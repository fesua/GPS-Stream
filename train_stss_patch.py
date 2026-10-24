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
from lib.temporal_gps import TemporalGPS, R
from lib.temporal_loader import ClipDataset, change_target
from lib.patch_step3 import PatchStep3
from train_temporal import to_cuda

# Step 2 trained for the patch pipeline: only the STSS head learns, inside the step-3 state dynamics (per-frame recipe
# inside P x P tiles around the cells STSS marks, previous values elsewhere; each cell's reference features = the frame
# of its last update). Target = the image-difference target of the method (cell colour change > 8/255 against the cell's
# reference), BCE with pos_weight 10 as in train_temporal.py. The first --target_mask_frac of training drives the state
# with the target mask, the rest with the STSS prediction (so the references evolve as at test time).
# python train_stss_patch.py --exp stss_patch64 --patch 64 --steps 4000
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'
CK = 'experiments/temporal_r8_ref/ckpt/step8000.pth'
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True); p.add_argument('--patch', type=int, default=64); p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--init', default=CK, help='TemporalGPS checkpoint whose STSS is the starting point')
    p.add_argument('--target_mask_frac', type=float, default=0.3); p.add_argument('--data', default='./GPS_plus_data/temporal'); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    out_dir = Path('experiments/%s/ckpt' % a.exp); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('experiments/%s/train.log' % a.exp), logging.StreamHandler()])
    cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost()
    cfg.dataset.local_data_root = a.data; cfg.dataset.train_data_root = os.path.join(a.data, 'train'); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(TEACHER, map_location='cuda')['network'])
    ck = torch.load(a.init, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1), full_attention=ck.get('full_attention', False), temporal_attention=ck.get('temporal_attention')).cuda()
    model.load_state_dict(ck['model'], strict=False); model.eval()
    for n, prm in model.named_parameters():
        prm.requires_grad = n.startswith('stss.')
    model.stss.train()
    runner = PatchStep3(model, a.patch, 'stss')
    params = [prm for prm in model.stss.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, a.steps + 100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    def forever(T):
        loader = DataLoader(ClipDataset(cfg.dataset, T), batch_size=1, shuffle=True, num_workers=a.workers, pin_memory=True)
        while True:
            yield from loader
    loaders = {T: forever(T) for T in (2, 3)}
    logging.info('STSS params %.3fM, patch %d px, %d steps' % (sum(q.numel() for q in params) / 1e6, a.patch, a.steps))
    log, t0 = [], time.time()
    for it in range(a.steps):
        T = random.choice((2, 3))
        clip = [to_cuda(d) for d in next(loaders[T])]
        with torch.no_grad():
            state = runner.keyframe(clip[0])
        for k in range(1, T):
            inp = model.inputs(clip[k])
            target = change_target(inp['image'], state['image'])
            with torch.no_grad():
                teach = model.keyframe(clip[k])
            logits = model.stss(teach['f3'], state['f3'])      # gradient only here
            if k == T - 1:
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(10.0, device=logits.device))
            with torch.no_grad():  # advance the state: target mask early in training, the prediction later (as at test time)
                changed = target if it < a.target_mask_frac * a.steps else (torch.sigmoid(logits) > 0.5).float()
                c = runner.c
                M = F.interpolate(F.max_pool2d(changed, c, c), scale_factor=c, mode='nearest') > 0.5 if c > 1 else changed > 0.5
                Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
                gn, gp = teach['g'], state['g']
                g = {'xyz': torch.where(Mf.flatten(1)[..., None], gn['xyz'], gp['xyz']), 'valid': gn['valid']}
                for kk in ('rot', 'scale', 'opacity', 'img', 'depth'):
                    g[kk] = torch.where(Mf, gn[kk], gp[kk])
                state = {'f3': torch.where(M, teach['f3'], state['f3']), 'image': torch.where(Mf, inp['image'], state['image']), 'flow': torch.where(Mf, teach['flow'], state['flow']), 'g': g}
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            pm = (torch.sigmoid(logits) > 0.5).float()
            log.append([loss.item(), (pm * target).sum().item() / max(target.sum().item(), 1), pm.mean().item(), target.mean().item()])
        if (it + 1) % 100 == 0:
            m = np.mean(log, 0); log = []
            logging.info('it %d  bce %.4f  recall %.3f  pred %.3f%%  target %.3f%%  %.1f min' % (it + 1, m[0], m[1], 100 * m[2], 100 * m[3], (time.time() - t0) / 60))
        if (it + 1) % 1000 == 0 or it + 1 == a.steps:
            torch.save({'model': model.state_dict(), 'step': it + 1, 'radius': ck.get('radius', R), 'update_dilation': ck.get('update_dilation', 1), 'patch': a.patch, 'stss_only': True}, out_dir / ('step%d.pth' % (it + 1)))
    logging.info('FINISHED')
