import argparse
import copy
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.temporal_gps import TemporalGPS, R
from lib.temporal_sparse import crop_tiles
from lib.temporal_patch import patch_tiles
from lib.patch_sparse import PatchSparse
from lib.trt_sparse import fold, Front, CropFront, DenseGRU, GRUStage, Regr, build_engine, TrtNet
from core.utils.utils import downflow8
from export_trt import timeit, source_inputs
from train_temporal import to_cuda

# TensorRT cost of the tile-cell execution, stage by stage (lib/trt_sparse.py): dense engines (both views) vs crop engines
# (batch = N tile crops, dynamic) as a function of N, on the tiles the pipeline really marks (cook_spinach s2). Also the
# fp16 difference of each engine to its PyTorch fp32 module on the same inputs. --build: export + build the engines
# (can share the GPU); --time: timings (GPU alone).
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'
OUT = 'experiments/gps_plus_dynerf_jit2_1007/trt/sparse'
p = argparse.ArgumentParser(); p.add_argument('--scene', default='cook_spinach'); p.add_argument('--set', type=int, default=2); p.add_argument('--frames', type=int, nargs=2, default=[221, 261])
p.add_argument('--ckpt', default='experiments/stss_patch64/ckpt/step4000.pth'); p.add_argument('--patch', type=int, default=128); p.add_argument('--halo', type=int, default=4)
p.add_argument('--reg_margin', type=int, default=4, help='regressor crop margin in cells (1/8)'); p.add_argument('--precision', default='fp16', choices=['fp16', 'fp32'])
p.add_argument('--build', action='store_true'); p.add_argument('--time', action='store_true'); p.add_argument('--ns', type=int, nargs='+', default=[1, 4, 8, 16, 32, 64, 128])
p.add_argument('--opt', type=int, default=32); p.add_argument('--max', type=int, default=128); p.add_argument('--workspace', type=int, default=4, help='GB'); p.add_argument('--only', nargs='+', default=None, help='restrict --build / --time to these engines'); a = p.parse_args()
cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
k, halo, mr = a.patch // 8, a.halo, a.reg_margin; tt = k + 2 * halo; S = 8 * (k + 2 * mr); fp16 = a.precision == 'fp16'
os.makedirs(OUT, exist_ok=True)
def load():
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(TEACHER, map_location='cuda')['network'])
    ck = torch.load(a.ckpt, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model.load_state_dict(ck['model'], strict=False)
    return model.eval()
ds = StereoHumanDataset(cfg.dataset, phase=a.scene + '_process'); idx_of = {n: i for i, n in enumerate(ds.sample_list)}
get = lambda t: to_cuda(torch.utils.data.default_collate([ds.get_item(idx_of['%s_s%d_%04d' % (a.scene, a.set, t)], novel_id=[2])]))
eng = lambda name: os.path.join(OUT, '%s_%s.engine' % (name, a.precision))
onnx_of = lambda name: os.path.join(OUT, '%s.onnx' % name)

with torch.no_grad():
    model = load(); sp = PatchSparse(model, a.patch, halo, dense_reg=True, reg_crop=mr)   # crop regressor: no DeltaCNN needed to count tiles / build engines
    s = sp.keyframe(get(220)); Ns = []
    for t in range(*a.frames):
        o, s = sp.step(get(t), s); Ns.append(patch_tiles(o['mask'], k).shape[0])
    probe = list(range(*a.frames))[int(np.argmax(Ns))]
    print('tiles %d px, halo %d: %.1f tiles per frame on average over frames %d-%d (both views; min %d, max %d at frame %d); crop %d x %d cells = %.1f%% of the cell map per frame' % (
        a.patch, halo, np.mean(Ns), a.frames[0], a.frames[1] - 1, min(Ns), max(Ns), probe, tt, tt, 100 * np.mean(Ns) * tt * tt / (2 * 128 * 128)), flush=True)
    N_act = int(round(np.mean(Ns)))
    del sp, model, s, o; torch.cuda.empty_cache()
    # the probe frame: state after the step, tiles, crops
    model = load(); sp = PatchSparse(model, a.patch, halo, dense_reg=True, reg_crop=mr); d = get(220); s = sp.keyframe(d)
    for t in range(a.frames[0], probe + 1):
        d = get(t); o, s = sp.step(d, s)
    idx = patch_tiles(o['mask'], k); n = idx.shape[0]
    inp = model.inputs(d); t_ = model.teacher
    sp._modes('live'); feat = t_.img_encoder(inp['image'])
    depth = model.to_depth(s['flow'], inp)
    fi8 = downflow8(inp['flow_init'])
    fo = F.pad(s['fmap'], (0, 0, halo, halo)); ar = torch.arange(tt, device='cuda')
    Y = (idx[:, 1] * k)[:, None] + ar[None, :]
    rows = fo[(1 - idx[:, 0])[:, None], :, Y].permute(0, 2, 1, 3).contiguous()
    crops = {'att2': crop_tiles(s['att2'], idx, k, halo), 'rows': rows, 'c0x': crop_tiles(sp.coords0[:, :1], idx, k, halo), 'fi': crop_tiles(fi8, idx, k, halo),
             'img': crop_tiles(inp['image'], idx, 8 * k, 8 * mr), 'dep': crop_tiles(depth, idx, 8 * k, 8 * mr), 'f1': crop_tiles(feat[0], idx, 4 * k, 4 * mr),
             'f2': crop_tiles(feat[1], idx, 2 * k, 2 * mr), 'f3': crop_tiles(s['feat_cs'], idx, k, mr)}
    # PyTorch fp32 modules (dense: live statistics as the recipe; crops: start-frame statistics as fixed affines)
    td = fold(copy.deepcopy(t_), frozen=False).eval(); tc = fold(copy.deepcopy(t_), frozen=True).eval()
    rs = t_.raft_stereo
    mods = {'enc': (td.img_encoder, (inp['image'],), ['f1', 'f2', 'f3']),
            'front': (Front(td).cuda(), (inp['image'],), ['f1', 'f2', 'f3', 'att1', 'att2', 'fcs', 'fm', 'net', 'ctx']),
            'gru': (DenseGRU(td.raft_stereo, 8).cuda(), None, ['up']),
            'regr': (Regr(td).cuda(), (inp['image'], depth, feat[0], feat[1], s['feat_cs']), ['rot', 'scale', 'opacity', 'res']),
            'front_crop%d' % tt: (CropFront(tc).cuda(), (crops['att2'],), ['fcs', 'fm', 'net', 'ctx']),
            'gru_crop%d' % tt: (GRUStage(tc.raft_stereo, 8, tt, tt).cuda(), None, ['up']),
            'regr_crop%d' % S: (Regr(tc).cuda(), (crops['img'], crops['dep'], crops['f1'], crops['f2'], crops['f3']), ['rot', 'scale', 'opacity', 'res'])}
    # the dense GRU's real start state and context come from the front module; the crop GRU's inputs from the crop front
    f1, f2, f3, att1, att2, fcs, fm, net, ctx = mods['front'][0](inp['image'])
    mods['gru'] = (mods['gru'][0], (fm, net, ctx, fi8), ['up'])
    fcs_p, fm_p, net_p, ctx_p = mods['front_crop%d' % tt][0](crops['att2'])
    mods['gru_crop%d' % tt] = (mods['gru_crop%d' % tt][0], (fm_p, rows, net_p, ctx_p, crops['c0x'], crops['fi']), ['up'])
    in_names = {'enc': ['image'], 'front': ['image'], 'gru': ['fm', 'net', 'ctx', 'fi'], 'regr': ['image', 'depth', 'f1', 'f2', 'fcs'],
                'front_crop%d' % tt: ['att2'], 'gru_crop%d' % tt: ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'], 'regr_crop%d' % S: ['img', 'dep', 'f1', 'f2', 'f3']}
    if a.build:
        for name, (mod, args, outs) in mods.items():
            if a.only and name not in a.only: continue
            dyn = 'crop' in name
            if not os.path.exists(onnx_of(name)):
                dynamic_axes = {nm: {0: 'N'} for nm in in_names[name] + outs} if dyn else None
                torch.onnx.export(mod, args, onnx_of(name), opset_version=17, input_names=in_names[name], output_names=outs, dynamic_axes=dynamic_axes)
                print('exported %s (sample batch %d)' % (name, args[0].shape[0]), flush=True)
            if not os.path.exists(eng(name)):
                prof = {nm: ((1,) + tuple(x.shape[1:]), (a.opt,) + tuple(x.shape[1:]), (a.max,) + tuple(x.shape[1:])) for nm, x in zip(in_names[name], args)} if dyn else None
                import time; t0 = time.time(); build_engine(onnx_of(name), eng(name), fp16, prof, a.workspace)
                print('built %s in %.0f s' % (eng(name), time.time() - t0), flush=True)
    if a.time:
        nets = {name: TrtNet(eng(name), in_names[name], mods[name][2]) for name in mods}
        tick = lambda: (torch.cuda.synchronize(), __import__('time').perf_counter())[1]
        # references
        ts = []
        for t in range(221, 231):
            dd = get(t); t0 = tick(); model.keyframe(dd); ts.append(tick() - t0)
        print('dense per-frame recipe, PyTorch fp32 (encoder..regressor): %.1f ms' % (1000 * np.mean(ts[2:])))
        f3_prev = s['f3']
        print('STSS, PyTorch (1/8 map, both views): %.2f ms' % timeit(lambda: torch.sigmoid(model.stss(f3, f3_prev)) > 0.5))
        Mf = F.interpolate(o['mask'].float(), scale_factor=8, mode='nearest') > 0.5; g = s['g']
        print('paste of the Gaussian maps (torch.where on 6 full-size maps): %.2f ms' % timeit(lambda: [torch.where(Mf, g[kk], g[kk]) for kk in ('rot', 'scale', 'opacity', 'depth', 'img')] + [torch.where(Mf.flatten(1)[..., None], g['xyz'], g['xyz'])]))
        full = 'experiments/gps_plus_dynerf_jit2_1007/trt/source_net_f3_it8_%s.engine' % a.precision
        if os.path.exists(full):
            fin = source_inputs(d); fnet = TrtNet(full, list(fin)); fargs = [fin[nm] for nm in fnet.inputs]
            print('per-frame network as one engine (%s, with depth + f3 outputs): %.1f ms' % (a.precision, timeit(lambda: fnet(*fargs))))
        dense = {}
        dense_pt = {}
        for name in ('enc', 'front', 'gru', 'regr'):
            mod, args, outs = mods[name]
            dense[name] = timeit(lambda: nets[name](*args))
            dense_pt[name] = timeit(lambda: mod(*args), n=20, warm=5)   # the same stage in PyTorch fp32
            ref = mod(*args); ref = ref if isinstance(ref, (tuple, list)) else [ref]; got = nets[name](*args)
            if name == 'gru': err = 'up |diff| %.3f px' % (got[0] - ref[0]).abs().mean().item()
            elif name == 'regr': err = 'opacity |diff| %.4f, res-depth |diff| %.5f' % ((got[2] - ref[2]).abs().mean().item(), (got[3] - ref[3]).abs().mean().item())
            else: err = 'last output |diff| %.4f (rms %.3f)' % ((got[-1] - ref[-1]).abs().mean().item(), ref[-1].pow(2).mean().sqrt().item())
            print('dense %s (%s, both views): %.1f ms | vs PyTorch fp32: %s' % (name, a.precision, dense[name], err), flush=True)
        print('dense stages: encoder %.1f + attention/resid/cnet %.1f + GRU x8 + upsample %.1f + regressor %.1f = %.1f ms' % (dense['enc'], dense['front'] - dense['enc'], dense['gru'], dense['regr'], dense['front'] + dense['gru'] + dense['regr']))
        print('dense stages PyTorch fp32: encoder %.1f + attention/resid/cnet %.1f + GRU x8 + upsample %.1f + regressor %.1f = %.1f ms' % (dense_pt['enc'], dense_pt['front'] - dense_pt['enc'], dense_pt['gru'], dense_pt['regr'], dense_pt['front'] + dense_pt['gru'] + dense_pt['regr']))
        # crop engines vs N
        names_c = ['front_crop%d' % tt, 'gru_crop%d' % tt, 'regr_crop%d' % S]
        def take(x, N):
            rep = int(math.ceil(N / x.shape[0])); return x.repeat(rep, *([1] * (x.dim() - 1)))[:N].contiguous()
        for name in names_c:
            mod, args, outs = mods[name]; ref = mod(*args); ref = ref if isinstance(ref, (tuple, list)) else [ref]; got = nets[name](*args)
            core = (slice(None), slice(None), slice(8 * halo, 8 * (halo + k)), slice(8 * halo, 8 * (halo + k)))
            if 'gru' in name: err = 'up |diff| %.3f px in the tile cores' % (got[0] - ref[0])[core].abs().mean().item()
            elif 'regr' in name: err = 'opacity |diff| %.4f' % (got[2] - ref[2]).abs().mean().item()
            else: err = 'ctx |diff| %.4f' % (got[-1] - ref[-1]).abs().mean().item()
            print('crop %s (%s) at N = %d (the probe frame): vs PyTorch fp32 with the same inputs: %s' % (name, a.precision, n, err), flush=True)
        Ns_t = sorted(set([N for N in a.ns if N <= a.max] + [N_act]))
        print('crop engines (%s), ms per call vs the number of tile crops N (tile %d px, halo %d -> %d x %d cells; regressor crop %d px):' % (a.precision, a.patch, halo, tt, tt, S))
        print('  %5s %8s | %12s %12s %12s | %8s' % ('N', 'cells%', 'front_crop', 'gru_crop', 'regr_crop', 'sum'))
        for N in Ns_t:
            row = []
            for name in names_c:
                args = [take(x, N) for x in mods[name][1]]
                row.append(timeit(lambda: nets[name](*args)))
            tag = ' <- mean tiles per frame' if N == N_act else ''
            print('  %5d %7.1f%% | %12.1f %12.1f %12.1f | %8.1f%s' % (N, 100 * N * tt * tt / (2 * 128 * 128), row[0], row[1], row[2], sum(row), tag), flush=True)
