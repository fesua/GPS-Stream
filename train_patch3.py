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
from lib.temporal_gps import TemporalGPS, R, fill_render_data
from lib.temporal_loader import ClipDataset
from train_temporal import to_cuda

# Step 3 trained: the per-frame network (GPS-Gaussian+ recipe: full attention, cost volume, fresh GRU state, 8 iterations,
# regressor) is fine-tuned INSIDE the patch pipeline - on clips, frame 0 = per-frame result, later frames: STSS (fixed,
# from --stss) marks changed cells, the P x P tiles around them take the per-frame network's values, the rest is kept;
# the loss is the novel-view render loss (0.8 L1 + 0.2 (1 - SSIM)) + 0.005 chamfer of the last frame, as in the baseline
# fine-tunes (lr 2e-5, 8000 steps, clips 2-4). The encoder stays frozen (STSS was trained on its features).
# python train_patch3.py --exp patch3_ft --patch 64 --stss experiments/stss_patch64/ckpt/step4000.pth
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'


def chamfer(g, n=10000):
    xyz, valid = g['xyz'], g['valid']
    pts = []
    for v in (0, 1):
        p = xyz[v][valid[v]]
        pts.append(p[torch.randperm(p.shape[0], device=p.device)[:n]])
    d = torch.cdist(pts[0], pts[1])
    return d.min(1)[0].mean() + d.min(0)[0].mean()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True); p.add_argument('--patch', type=int, default=64); p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--lr', type=float, default=2e-5); p.add_argument('--chamfer', type=float, default=0.005); p.add_argument('--clip_lens', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--stss', default='experiments/stss_patch64/ckpt/step4000.pth'); p.add_argument('--teacher', default=TEACHER)
    p.add_argument('--data', default='./GPS_plus_data/temporal'); p.add_argument('--workers', type=int, default=6); p.add_argument('--seed', type=int, default=0); p.add_argument('--log_every', type=int, default=100)
    a = p.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    out_dir = Path('experiments/%s/ckpt' % a.exp); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('experiments/%s/train.log' % a.exp), logging.StreamHandler()])
    cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost()
    cfg.dataset.local_data_root = a.data; cfg.dataset.train_data_root = os.path.join(a.data, 'train'); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(a.teacher, map_location='cuda')['network'])
    ck = torch.load(a.stss, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1), full_attention=ck.get('full_attention', False), temporal_attention=ck.get('temporal_attention')).cuda()
    model.load_state_dict(ck['model'], strict=False); model.eval()
    # the clip's first frame comes from the FROZEN original per-frame network (as the start frame does at test time and as
    # the baseline fine-tunes did); only the tiles of later frames come from the network being trained
    teacher0 = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network'])
    model0 = TemporalGPS(teacher0, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model0.load_state_dict(ck['model'], strict=False)
    teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network']); model0.eval()
    for prm in model0.parameters():
        prm.requires_grad = False
    for prm in model.parameters():
        prm.requires_grad = False
    for n, prm in teacher.named_parameters():      # the per-frame network learns, except the encoder (STSS input)
        prm.requires_grad = not n.startswith('img_encoder.')
    teacher.train()
    params = [prm for prm in teacher.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-5, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, a.steps + 100, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    def forever(T):
        loader = DataLoader(ClipDataset(cfg.dataset, T), batch_size=1, shuffle=True, num_workers=a.workers, pin_memory=True)
        while True:
            yield from loader
    loaders = {T: forever(T) for T in a.clip_lens}
    c = a.patch // 8
    logging.info('trainable params %.2fM (encoder frozen; clip frame 0 from the frozen original network), patch %d px, %d steps, clips %s' % (sum(q.numel() for q in params) / 1e6, a.patch, a.steps, a.clip_lens))
    log, t0 = [], time.time()
    for it in range(a.steps):
        T = random.choice(a.clip_lens)
        clip = [to_cuda(d) for d in next(loaders[T])]
        with torch.no_grad():
            st = model0.keyframe(clip[0]); state = {'f3': st['f3'], 'image': st['image'], 'flow': st['flow'], 'g': st['g']}
        for k in range(1, T):
            last = k == T - 1
            inp = model.inputs(clip[k])
            if last:   # per-frame recipe WITH gradients: TemporalGPS.keyframe is decorated @torch.no_grad, call the raw function
                teach = TemporalGPS.keyframe.__wrapped__(model, clip[k])
            else:
                teach = model.keyframe(clip[k])
            with torch.no_grad():
                changed = (torch.sigmoid(model.stss(teach['f3'].detach(), state['f3'])) > 0.5).float()
                M = F.interpolate(F.max_pool2d(changed, c, c), scale_factor=c, mode='nearest') > 0.5 if c > 1 else changed > 0.5
                Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
            gn, gp = teach['g'], state['g']
            g = {'xyz': torch.where(Mf.flatten(1)[..., None], gn['xyz'], gp['xyz']), 'valid': gn['valid']}
            for kk in ('rot', 'scale', 'opacity', 'img', 'depth'):
                g[kk] = torch.where(Mf, gn[kk], gp[kk])
            if last:
                out_g, mask_frac = g, M.float().mean().item()
            else:
                state = {'f3': torch.where(M, teach['f3'], state['f3']).detach(), 'image': torch.where(Mf, inp['image'], state['image']),
                         'flow': torch.where(Mf, teach['flow'], state['flow']).detach(), 'g': {kk: v.detach() for kk, v in g.items()}}
        data = fill_render_data(clip[-1], out_g)
        pred = pts2render(data, bg_color=cfg.dataset.bg_color)['novel_view']['img_pred']
        gt = data['novel_view']['img'].cuda()
        l_render = 0.8 * l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
        l_cham = chamfer(out_g) if a.chamfer > 0 else torch.zeros((), device=pred.device)
        loss = l_render + a.chamfer * l_cham
        assert loss.requires_grad, 'no gradient path: the per-frame recipe ran under no_grad'
        opt.zero_grad()
        gnorm = torch.zeros(())
        if torch.isfinite(loss):
            loss.backward(); gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        with torch.no_grad():
            log.append([l_render.item(), l_cham.item(), (10 * torch.log10(1 / ((pred - gt) ** 2).mean())).item(), mask_frac, T, float(gnorm)])
        if (it + 1) % a.log_every == 0:
            m = np.mean(log, 0); log = []
            logging.info('it %d  render %.4f chamfer %.4f psnr %.2f | updated %.3f T %.2f | grad %.3f | lr %.2e | %.2f s/it' % (it + 1, m[0], m[1], m[2], m[3], m[4], m[5], sched.get_last_lr()[0], (time.time() - t0) / (it + 1)))
        if (it + 1) % 2000 == 0 or it + 1 == a.steps:
            torch.save({'network': teacher.state_dict(), 'step': it + 1, 'patch': a.patch, 'stss': a.stss}, out_dir / ('step%d.pth' % (it + 1)))
    logging.info('FINISHED')
