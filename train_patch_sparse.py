import os
import time
import random
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import RtStereoHumanModel
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.temporal_gps import TemporalGPS, R, fill_render_data
from lib.temporal_loader import ClipDataset
from lib.patch_sparse import PatchSparse
from train_patch3 import chamfer, TEACHER
from train_temporal import to_cuda

# The tile-cell execution (lib/patch_sparse.py: query-sparse attention, crop-batched residual block / cnet / GRU x 8 /
# upsampling with halo cells, regressor on the patchwork) TRAINED under its own condition: the per-frame network is
# fine-tuned on clips - frame 0 = dense per-frame result of the FROZEN original network (as the start frame at test time),
# later frames run the sparse step with the network being trained (no gradient), the last frame runs it WITH gradients;
# loss = novel-view render loss (0.8 L1 + 0.2 (1 - SSIM)) + 0.005 chamfer, lr 2e-5, 8000 steps, clips 2-4, encoder frozen
# (STSS input) - the same recipe as train_patch3.py. The regressor runs as the dense module on the patchwork (same
# arithmetic as the DeltaCNN used at test time, differentiable).
# python train_patch_sparse.py --exp patch_sparse_ft128h4 --patch 128 --halo 4

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True); p.add_argument('--patch', type=int, default=128); p.add_argument('--halo', type=int, default=4); p.add_argument('--steps', type=int, default=8000); p.add_argument('--reg_crop', type=int, default=None, help='regressor on tile crops with this margin in cells (0 = the bare tile); None = dense on the patchwork')
    p.add_argument('--depth_mode', default='sparse', choices=['sparse', 'dense'], help='sparse: attention/cnet/GRU on the tile crops; dense: per-frame recipe every frame (variant B)')
    p.add_argument('--reg_mode', default=None, choices=[None, 'crop', 'dense_pw', 'dense_all'], help='dense_all: regressor dense on the whole frame, output used everywhere (variant A)')
    p.add_argument('--tile_ctx', action='store_true', help='variant D: stored-context decoding (lib/tile_mode.py), use with --halo 0 --reg_crop 0')
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
    teacher0 = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network'])
    model0 = TemporalGPS(teacher0, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model0.load_state_dict(ck['model'], strict=False)
    teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network']); model0.eval()
    for prm in list(model0.parameters()) + list(model.parameters()):
        prm.requires_grad = False
    for n, prm in teacher.named_parameters():      # the per-frame network learns, except the encoder (STSS input)
        prm.requires_grad = not n.startswith('img_encoder.')
    params = [prm for prm in teacher.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-5, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, a.steps + 100, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    reg_mode = a.reg_mode or ('crop' if a.reg_crop is not None else 'dense_pw')
    sp = PatchSparse(model, a.patch, a.halo, key_model=model0, dense_reg=True, reg_crop=a.reg_crop, depth_sparse=a.depth_mode == 'sparse', reg_mode=reg_mode, tile_ctx=a.tile_ctx)   # frozen norms installed in place in `teacher`
    def forever(T):
        loader = DataLoader(ClipDataset(cfg.dataset, T), batch_size=1, shuffle=True, num_workers=a.workers, pin_memory=True)
        while True:
            yield from loader
    loaders = {T: forever(T) for T in a.clip_lens}
    logging.info('tile-cell execution trained: trainable params %.2fM (encoder frozen; clip frame 0 and stored maps from the frozen original network), tiles %d px, halo %d cells, depth %s, regressor %s, %d steps, clips %s' % (
        sum(q.numel() for q in params) / 1e6, a.patch, a.halo, a.depth_mode + (' + stored-context decoding' if a.tile_ctx else ''), reg_mode + ('' if reg_mode != 'crop' else ' (margin %d cells)' % a.reg_crop), a.steps, a.clip_lens))
    log, t0 = [], time.time()
    for it in range(a.steps):
        T = random.choice(a.clip_lens)
        clip = [to_cuda(d) for d in next(loaders[T])]
        state = sp.keyframe(clip[0])                               # no grad: frozen start frame, statistics of the current weights
        for k in range(1, T - 1):
            out, state = sp.step(clip[k], state)                   # no grad, current weights
        out, _ = PatchSparse.step.__wrapped__(sp, clip[T - 1], state)   # the last frame WITH gradients (step is decorated @torch.no_grad)
        data = fill_render_data(clip[-1], out['g'])
        pred = pts2render(data, bg_color=cfg.dataset.bg_color)['novel_view']['img_pred']
        gt = data['novel_view']['img'].cuda()
        l_render = 0.8 * l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
        l_cham = chamfer(out['g']) if a.chamfer > 0 else torch.zeros((), device=pred.device)
        loss = l_render + a.chamfer * l_cham
        assert loss.requires_grad, 'no gradient path through the sparse step'
        opt.zero_grad()
        gnorm = torch.zeros(())
        if torch.isfinite(loss):
            loss.backward(); gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        with torch.no_grad():
            log.append([l_render.item(), l_cham.item(), (10 * torch.log10(1 / ((pred - gt) ** 2).mean())).item(), out['mask'].mean().item(), out['crop_cells'] / (2 * 128 * 128), T, float(gnorm)])
        if (it + 1) % a.log_every == 0:
            m = np.mean(log, 0); log = []
            logging.info('it %d  render %.4f chamfer %.4f psnr %.2f | updated %.3f crop cells %.3f T %.2f | grad %.3f | lr %.2e | %.2f s/it' % (it + 1, m[0], m[1], m[2], m[3], m[4], m[5], m[6], sched.get_last_lr()[0], (time.time() - t0) / (it + 1)))
        if (it + 1) % 2000 == 0 or it + 1 == a.steps:
            torch.save({'network': teacher.state_dict(), 'step': it + 1, 'patch': a.patch, 'halo': a.halo, 'reg_crop': a.reg_crop, 'depth_mode': a.depth_mode, 'reg_mode': reg_mode, 'tile_ctx': a.tile_ctx, 'stss': a.stss}, out_dir / ('step%d.pth' % (it + 1)))
    logging.info('FINISHED')
