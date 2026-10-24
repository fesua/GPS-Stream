import os
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import RtStereoHumanModel
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.temporal_gps import TemporalGPS, R, detach_state, fill_render_data, border_depth_loss
from lib.temporal_sparse import freeze_norms
from lib.temporal_loader import ClipDataset, change_target

# python train_temporal.py --teacher experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth
# Clips of 2-3 consecutive frames: the first frame is a keyframe of the per-frame model (teacher), the
# following frames go through TemporalGPS; gradients flow through the last frame only.


def to_cuda(data):
    for view in ['lmain', 'rmain']:
        data[view] = {k: v.cuda() for k, v in data[view].items()}
    for k in ['world_view_transform', 'full_proj_transform', 'camera_center']:
        data['novel_view'][k] = data['novel_view'][k].cuda()
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True)
    parser.add_argument('--data', type=str, default='./GPS_plus_data/temporal')
    parser.add_argument('--exp', type=str, default='temporal_gps')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=5e-5, help='modules copied from the per-frame model')
    parser.add_argument('--lr_new', type=float, default=1e-3, help='new modules: STSS, proposal MLP, GRU cost-volume input')
    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--radius', type=int, default=R, help='local epipolar window +- radius (feature cells)')
    parser.add_argument('--update_dilation', type=int, default=1, help='cells added around the STSS changes (0: STSS cells only)')
    parser.add_argument('--target_mask_frac', type=float, default=0.3, help='first part of training uses the target change mask')
    parser.add_argument('--border_weight', type=float, default=0.0, help='depth mismatch on the border between updated and kept pixels')
    parser.add_argument('--border_gamma', type=float, default=10.0, help='colour-edge weighting of the border loss')
    parser.add_argument('--temporal_attention', default=None, choices=['mix', 'only'], help='with --full_attention')
    parser.add_argument('--full_attention', action='store_true', help='original epipolar attention + cnet on the whole map, no proposal MLP')
    parser.add_argument('--patch', type=int, default=0, help='>0: conv parts on patches of this many cells (lib/temporal_patch.py)')
    parser.add_argument('--halo', type=int, default=0, help='patch mode: context cells around each patch')
    arg = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost()
    cfg.dataset.local_data_root = arg.data
    cfg.dataset.train_data_root = os.path.join(arg.data, 'train')
    cfg.dataset.inverse_depth_init = 0.5
    cfg.raft.val_iters = cfg.raft.train_iters = arg.iters
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda()
    teacher.load_state_dict(torch.load(arg.teacher, map_location='cuda')['network'])
    model = TemporalGPS(teacher, iters=arg.iters, radius=arg.radius, update_dilation=arg.update_dilation,
                        full_attention=arg.full_attention, temporal_attention=arg.temporal_attention).cuda().train()
    if arg.patch:  # patches use the full-map normalisation statistics recorded at the clip's first frame
        for mod in [model.att_resid, model.cnet, model.regressor]:
            freeze_norms(mod)

    def forever(T):
        loader = DataLoader(ClipDataset(cfg.dataset, T), batch_size=1, shuffle=True, num_workers=6, pin_memory=True)
        while True:
            yield from loader
    loaders = {T: forever(T) for T in [2, 3]}
    new_modules = ('stss.', 'proposal.', 'update_block.encoder.convc1.', 'temporal_att.')
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = [{'params': [p for n, p in named if not n.startswith(new_modules)], 'lr': arg.lr},
              {'params': [p for n, p in named if n.startswith(new_modules)], 'lr': arg.lr_new}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-5, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [arg.lr, arg.lr_new], arg.steps + 100, pct_start=0.01,
                                                cycle_momentum=False, anneal_strategy='linear')
    out_dir = Path('experiments/%s/ckpt' % arg.exp); out_dir.mkdir(parents=True, exist_ok=True)
    logging.info('trainable params %.2fM' % (sum(p.numel() for p in params) / 1e6))

    log = []
    t0 = time.time()
    for it in range(arg.steps):
        T = random.choice([2, 3])
        clip = [to_cuda(d) for d in next(loaders[T])]
        with torch.no_grad():
            teach = [model.keyframe(d) for d in clip]  # per-frame model: keyframe state and targets
        state = model.prepare_patch(clip[0], teach[0]) if arg.patch else teach[0]
        for k in range(1, T):
            target = change_target(model.inputs(clip[k])['image'], state['image'])  # vs the frame each cell was last updated in
            mask = target if it < arg.target_mask_frac * arg.steps else None
            last = k == T - 1
            with torch.set_grad_enabled(last):
                if arg.patch:
                    out, new_state = model.step_patch(clip[k], state, arg.patch, arg.halo, mask=mask)
                else:
                    out, new_state = model.step(clip[k], state, mask=mask)
            if not last:
                state = detach_state(new_state)

        data = fill_render_data(clip[-1], out['g'])
        pred = pts2render(data, bg_color=cfg.dataset.bg_color)['novel_view']['img_pred']
        gt = data['novel_view']['img'].cuda()
        l_render = 0.8 * l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
        M = out['mask']
        Mf = F.interpolate(M, scale_factor=8, mode='nearest')
        l_flow = ((out['flow'] - teach[-1]['flow']).abs() * Mf).sum() / Mf.sum().clamp(min=1)  # updated region only
        err = (out['p_star'] - teach[-1]['centers']).abs()
        l_prop = (F.smooth_l1_loss(out['p_star'], teach[-1]['centers'], reduction='none') * M).sum() / M.sum().clamp(min=1)
        l_stss = F.binary_cross_entropy_with_logits(out['logits'], target, pos_weight=torch.tensor(10.0, device=target.device))
        inp = model.inputs(clip[-1])
        l_border = border_depth_loss(out['g']['depth'], out['groups'] if arg.patch else Mf > 0.5, inp['image'],
                                     inp['mask'][:, :1] > 0.5, arg.border_gamma)
        loss = l_render + 0.01 * l_flow + 0.05 * l_prop + 0.1 * l_stss + arg.border_weight * l_border

        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

        with torch.no_grad():
            pm = (torch.sigmoid(out['logits']) > 0.5).float()
            recall = (pm * target).sum() / target.sum().clamp(min=1)
            sel = M > 0.5
            log.append([l_render.item(), l_flow.item(), l_prop.item(), l_stss.item(), M.mean().item(), target.mean().item(),
                        recall.item(), err[sel].mean().item() if sel.any() else 0, (err[sel] <= arg.radius).float().mean().item() if sel.any() else 1,
                        l_border.item()])
        if (it + 1) % 100 == 0:
            a = np.mean(log, 0); log = []
            logging.info('it %d  render %.4f flow %.3f prop %.3f stss %.3f border %.4f | mask %.3f target %.3f recall %.2f | '
                         'proposal err %.2f cells, within +-%d %.3f | %.2f s/it'
                         % (it + 1, *a[:4], a[9], *a[4:8], arg.radius, a[8], (time.time() - t0) / (it + 1)))
        if (it + 1) % 2000 == 0 or it + 1 == arg.steps:
            torch.save({'model': {k: v for k, v in model.state_dict().items() if not k.startswith('teacher.')}, 'step': it + 1, 'radius': arg.radius, 'update_dilation': arg.update_dilation,
                        'border_weight': arg.border_weight, 'patch': arg.patch, 'halo': arg.halo,
                        'full_attention': arg.full_attention, 'temporal_attention': arg.temporal_attention},
                       out_dir / ('step%d.pth' % (it + 1)))
    logging.info('FINISHED')
