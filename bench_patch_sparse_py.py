import argparse
import numpy as np
import torch
from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.temporal_gps import TemporalGPS, R, fill_render_data
from lib.GaussianRender import pts2render
from lib.patch_sparse import PatchSparse
from train_temporal import to_cuda

# per-frame time of the PyTorch tile-only execution for one variant (complete pipeline, state carried, render excluded; render timed separately)
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'
p = argparse.ArgumentParser(); p.add_argument('--scene', default='cook_spinach'); p.add_argument('--set', type=int, default=2); p.add_argument('--frames', type=int, nargs=2, default=[221, 261])
p.add_argument('--ckpt', default='experiments/stss_patch64/ckpt/step4000.pth'); p.add_argument('--teacher', required=True)
p.add_argument('--patch', type=int, default=128); p.add_argument('--halo', type=int, default=0); p.add_argument('--reg_cells', type=int, default=0)
p.add_argument('--depth_mode', default='sparse', choices=['sparse', 'dense']); p.add_argument('--reg_mode', default='crop', choices=['crop', 'dense_all', 'dense_pw', 'delta']); p.add_argument('--tile_ctx', action='store_true')
a = p.parse_args()
cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
def load(w):
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(w, map_location='cuda')['network'])
    ck = torch.load(a.ckpt, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model.load_state_dict(ck['model'], strict=False)
    teacher.load_state_dict(torch.load(w, map_location='cuda')['network']); return model.eval()
ds = StereoHumanDataset(cfg.dataset, phase=a.scene + '_process'); idx = {n: i for i, n in enumerate(ds.sample_list)}
get = lambda t: to_cuda(torch.utils.data.default_collate([ds.get_item(idx['%s_s%d_%04d' % (a.scene, a.set, t)], novel_id=[2])]))
tick = lambda: (torch.cuda.synchronize(), __import__('time').perf_counter())[1]
with torch.no_grad():
    key = load(TEACHER); model = load(a.teacher)
    sp = PatchSparse(model, a.patch, a.halo, key_model=key, dense_reg=a.reg_mode != 'delta', reg_crop=a.reg_cells if a.reg_mode == 'crop' else None, depth_sparse=a.depth_mode == 'sparse', reg_mode=a.reg_mode, tile_ctx=a.tile_ctx)
    s = sp.keyframe(get(220)); ts, tr, area, cells, stages = [], [], [], [], {}
    for t in range(*a.frames):
        d = get(t); t0 = tick(); o, s = sp.step(d, s); ts.append(tick() - t0)
        for kk, v in o['times'].items(): stages.setdefault(kk, []).append(v)
        area.append(o['mask'].mean().item()); cells.append(o['crop_cells'] / (2 * 128 * 128))
        dd = fill_render_data(d, o['g']); t0 = tick(); pts2render(dd, bg_color=[0, 0, 0]); tr.append(tick() - t0)
    print('PyTorch tile-only execution (%s): depth %s%s, regressor %s, tiles %d px, halo %d, regressor margin %d | area %.1f%%, crop cells %.1f%% | per frame (complete pipeline, render excluded) %.1f ms = %s | + render %.1f ms' % (
        a.teacher, a.depth_mode, ' + stored context' if a.tile_ctx else '', a.reg_mode, a.patch, a.halo, a.reg_cells, 100 * np.mean(area), 100 * np.mean(cells), 1000 * np.mean(ts[2:]),
        ', '.join('%s %.1f' % (kk, 1000 * np.mean(v[2:])) for kk, v in stages.items()), 1000 * np.mean(tr[2:])), flush=True)
