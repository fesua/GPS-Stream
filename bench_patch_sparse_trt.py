import argparse
import numpy as np
import torch
import torch.nn.functional as F
from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.temporal_gps import TemporalGPS, R, fill_render_data
from lib.GaussianRender import pts2render
from lib.patch_sparse import PatchSparse
from lib.patch_sparse_trt import PatchSparseTRT
from train_temporal import to_cuda

# Tile-cell execution with crop engines (lib/patch_sparse_trt.py): per-frame time of the COMPLETE pipeline (every frame,
# state carried, encoder engine -> STSS -> sparse attention -> crop engines -> tiles written back; render and data loading
# excluded, render timed separately) and the one-step difference to the PyTorch tile-cell execution with the same weights.
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'
p = argparse.ArgumentParser(); p.add_argument('--scene', default='cook_spinach'); p.add_argument('--set', type=int, default=2); p.add_argument('--frames', type=int, nargs=2, default=[221, 261])
p.add_argument('--ckpt', default='experiments/stss_patch64/ckpt/step4000.pth'); p.add_argument('--teacher', default='experiments/patch_sparse_ft128h4/ckpt/step8000.pth')
p.add_argument('--engine_dir', default='experiments/gps_plus_dynerf_jit2_1007/trt/sparse_ft128h4'); p.add_argument('--patch', type=int, default=128); p.add_argument('--halo', type=int, default=4); p.add_argument('--reg_margin', type=int, default=4)
p.add_argument('--depth_mode', default='sparse', choices=['sparse', 'dense']); p.add_argument('--reg_mode', default='crop', choices=['crop', 'dense_all'])
p.add_argument('--tile_ctx', action='store_true'); p.add_argument('--graph', action='store_true', help='crop / ring engines launched as CUDA graphs (per batch size, padded to 8)')
a = p.parse_args()
cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
def load(weights):
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(weights, map_location='cuda')['network'])
    ck = torch.load(a.ckpt, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model.load_state_dict(ck['model'], strict=False)
    teacher.load_state_dict(torch.load(weights, map_location='cuda')['network'])
    return model.eval()
ds = StereoHumanDataset(cfg.dataset, phase=a.scene + '_process'); idx = {n: i for i, n in enumerate(ds.sample_list)}
get = lambda t: to_cuda(torch.utils.data.default_collate([ds.get_item(idx['%s_s%d_%04d' % (a.scene, a.set, t)], novel_id=[2])]))
tick = lambda: (torch.cuda.synchronize(), __import__('time').perf_counter())[1]
with torch.no_grad():
    key = load(TEACHER)                                   # frozen original network: start frame and stored maps
    ds_ = a.depth_mode == 'sparse'
    model_t = load(a.teacher); trt = PatchSparseTRT(model_t, a.engine_dir, a.patch, a.halo, a.reg_margin, key_model=key, depth_sparse=ds_, reg_mode=a.reg_mode, tile_ctx=a.tile_ctx, graph=a.graph)
    model_p = load(a.teacher); ref = PatchSparse(model_p, a.patch, a.halo, key_model=key, dense_reg=True, reg_crop=a.reg_margin, depth_sparse=ds_, reg_mode=a.reg_mode, tile_ctx=a.tile_ctx)   # the PyTorch execution of the same variant (its own training condition)
    d0 = get(220); s_t = trt.keyframe(d0); s_r = ref.keyframe(d0)
    times, times_p, step_t, step_p, e_dep, e_flow, area, cells, tr = {}, {}, [], [], [], [], [], [], []
    for t in range(*a.frames):
        d = get(t)
        t0 = tick(); o_t, s_t = trt.step(d, s_t); step_t.append(tick() - t0)
        t0 = tick(); o_r, s_r = ref.step(d, s_r); step_p.append(tick() - t0)
        Mf = F.interpolate(o_t['mask'], scale_factor=8, mode='nearest') > 0.5; tf = d['lmain']['Tf_x'].abs().view(-1)[0]
        if Mf.any():
            e_flow.append((o_t['flow'] - o_r['flow']).abs()[Mf].mean().item())
            e_dep.append(((o_t['g']['depth'] - o_r['g']['depth']).abs() * tf)[Mf].mean().item())
        for kk, v in o_t['times'].items(): times.setdefault(kk, []).append(v)
        for kk, v in o_r['times'].items(): times_p.setdefault(kk, []).append(v)
        area.append(o_t['mask'].mean().item()); cells.append(o_t['crop_cells'] / (2 * 128 * 128))
        dd = fill_render_data(d, o_t['g']); t0 = tick(); pts2render(dd, bg_color=[0, 0, 0]); tr.append(tick() - t0)
        # keep both on the PyTorch pipeline's state so each step is compared on its own
        s_t['flow'] = s_r['flow'].clone(); s_t['g'] = {kk: v.clone() for kk, v in s_r['g'].items()}
        for kk in ('att1', 'att2', 'feat_cs', 'fmap'): s_t[kk] = s_r[kk].clone()
        s_t['ctx'] = [c.clone() for c in s_r['ctx']]; s_t['reg_in'] = tuple(x.clone() if torch.is_tensor(x) else [y.clone() for y in x] for x in s_r['reg_in'])
    stage = ', '.join('%s %.1f' % (kk, 1000 * np.mean(v[2:])) for kk, v in times.items())
    stage_p = ', '.join('%s %.1f' % (kk, 1000 * np.mean(v[2:])) for kk, v in times_p.items())
    print('tile-cell execution (%s): depth %s%s, regressor %s, tiles %d px, halo %d, regressor margin %d cells | area %.1f%%, crop cells %.1f%% | per frame (complete pipeline, render excluded): TensorRT %.1f ms = %s | PyTorch %.1f ms = %s | + render %.1f ms' % (
        a.teacher, a.depth_mode, ' + stored context' if a.tile_ctx else '', a.reg_mode + (' (CUDA graphs)' if a.graph else ''), a.patch, a.halo, a.reg_margin, 100 * np.mean(area), 100 * np.mean(cells), 1000 * np.mean(step_t[2:]), stage, 1000 * np.mean(step_p[2:]), stage_p, 1000 * np.mean(tr[2:])), flush=True)
