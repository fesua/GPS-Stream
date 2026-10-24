import argparse
import copy
import os
import time
import torch
from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import RtStereoHumanModel
from lib.temporal_sparse import freeze_norms
from lib.frozen_norm import freeze_group_norms
from lib.trt_sparse import fold_input_affine, CropFrontAff, GRUStage, RegrAff, RowAttn, build_engine, fold, Front, DenseGRU, Regr, GRUStageRing, RingWrap, RingWrapG, RingWrapB

# Crop engines of the tile-cell execution for a given per-frame network (e.g. fine-tuned under the crop condition):
# attention residual block + cnet, GRU x 8 + upsampling, regressor - batch = N tile crops (dynamic), normalisation statistics
# as an input (lib/trt_sparse.py). The encoder engine (frozen weights) is shared: trt/sparse/enc_fp16.engine.
# python build_sparse_trt.py --teacher experiments/patch_sparse_ft128h4/ckpt/step8000.pth --out trt/sparse_ft128h4
p = argparse.ArgumentParser(); p.add_argument('--teacher', required=True); p.add_argument('--out', required=True, help='engine directory under experiments/gps_plus_dynerf_jit2_1007/')
p.add_argument('--patch', type=int, default=128); p.add_argument('--halo', type=int, default=4); p.add_argument('--reg_margin', type=int, default=4)
p.add_argument('--precision', default='fp16'); p.add_argument('--opt', type=int, default=32); p.add_argument('--max', type=int, default=96); p.add_argument('--workspace', type=int, default=4)
p.add_argument('--dense', nargs='*', default=[], help="dense stage engines to build from these weights too: 'front' (encoder+attention+cnet), 'gru', 'regr'")
p.add_argument('--tile_ctx', action='store_true', help='variant D: ring-input engines (one ring per convolution / upsampling, from the per-layer stores; the tile inputs come back for the store write-back)')
p.add_argument('--ring_mode', default='band', choices=['layer', 'group', 'band'], help="variant D engines: 'layer' one ring per layer, 'group' one full ring window per (resolution, ring) group, 'band' the ring strips per group from band stores")
p.add_argument('--stss', default='experiments/stss_patch64/ckpt/step4000.pth'); a = p.parse_args()
cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
k, halo, mr = a.patch // 8, a.halo, a.reg_margin; tt = k + 2 * halo; S = 8 * (k + 2 * mr); fp16 = a.precision == 'fp16'
Wm = 128 + (-128) % k   # width of the padded 1/8 map (144 for 192 px tiles): rows of the cost volume and of the attention engines
out = os.path.join('experiments/gps_plus_dynerf_jit2_1007', a.out); os.makedirs(out, exist_ok=True)
t = RtStereoHumanModel(cfg, with_gs_render=True).cuda().eval(); t.load_state_dict(torch.load(a.teacher, map_location='cuda')['network'])
freeze_norms(t.loftr_coarse.resid); freeze_norms(t.raft_stereo.cnet); freeze_group_norms(t.gs_parm_regresser)   # the same wrappers / order as PatchSparse
tc = copy.deepcopy(t)
P_front = fold_input_affine(tc, [tc.loftr_coarse.resid, tc.raft_stereo.cnet]); P_regr = fold_input_affine(tc, [tc.gs_parm_regresser])
print('affine inputs: front P = %d, regressor P = %d' % (P_front, P_regr), flush=True)
N = 36; z = lambda *s: torch.randn(*s, device='cuda')
mods = {'front_crop%d' % tt: (CropFrontAff(tc).cuda(), (z(N, 96, tt, tt), z(N, P_front)), ['att2', 'aff'], ['fcs', 'fm', 'net', 'ctx']),
        'gru_crop%d' % tt: (GRUStage(tc.raft_stereo, 8, tt, tt).cuda(), (z(N, 192, tt, tt), z(N, 192, tt, Wm), z(N, 96, tt, tt), z(N, 288, tt, tt), z(N, 1, tt, tt), z(N, 1, tt, tt)), ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'], ['up']),
        'regr_crop%d' % S: (RegrAff(tc).cuda(), (z(N, 3, S, S), z(N, 1, S, S), z(N, 32, S // 2, S // 2), z(N, 48, S // 4, S // 4), z(N, 96, S // 8, S // 8), z(N, P_regr)), ['img', 'dep', 'f1', 'f2', 'f3', 'aff'], ['rot', 'scale', 'opacity', 'res']),
        'att_rows_l0': (RowAttn(tc.loftr_coarse.layers[0]).cuda(), (z(64, Wm, 96), z(64, Wm, 96)), ['x', 'src'], ['out']),
        'att_rows_l1': (RowAttn(tc.loftr_coarse.layers[1]).cuda(), (z(64, Wm, 96), z(64, Wm, 96)), ['x', 'src'], ['out'])}
if a.tile_ctx:   # replace the crop stages by ring-input versions; the ring shapes come from a record pass on a real frame
    from lib.temporal_gps import TemporalGPS, R
    from lib.patch_sparse import PatchSparse
    from lib.human_loader import StereoHumanDataset
    from train_temporal import to_cuda
    ck = torch.load(a.stss, map_location='cuda')
    model = TemporalGPS(t, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); model.load_state_dict(ck['model'], strict=False); model.eval()
    t.load_state_dict(torch.load(a.teacher, map_location='cuda')['network'])
    sp = PatchSparse(model, a.patch, a.halo, dense_reg=True, reg_crop=a.reg_margin, tile_ctx=True)
    ds = StereoHumanDataset(cfg.dataset, phase='cook_spinach_process'); i0 = ds.sample_list.index('cook_spinach_s2_0220')
    with torch.no_grad():
        sp.keyframe(to_cuda(torch.utils.data.default_collate([ds.get_item(i0, novel_id=[2])])))
    w8 = 128   # width of the 1/8 cell map the stores were recorded at (1024 px images)
    def rings(stage):
        out = []
        for key in sp.stage_keys[stage]:
            r, sw = sp.meta[key]; C = sp.stores[key].shape[1]; s_t = k * sw // w8
            out.append(z(N, C, s_t + 2 * r, s_t + 2 * r))
        return out
    base = mods
    import json
    if a.ring_mode == 'band':   # band stores: per group the ring strips (rows above / below, columns left / right) in, the tiles' border strips out
        from lib.tile_mode import ring_groups
        G = ring_groups(sp.stage_keys, sp.meta, sp.stores)
        def ringsb(st):
            out = []
            for g in G[st]:
                s_t, r = k * g['sw'] // w8, g['r']; out += [z(N, g['C'], 2 * r, s_t + 2 * r), z(N, g['C'], s_t, 2 * r)]
            return out
        rf, rg, rr = ringsb('front'), ringsb('gru'), ringsb('regr')
        print('band groups: front %d, gru %d, regr %d (layers %d, %d, %d)' % (len(rf) // 2, len(rg) // 2, len(rr) // 2, *[len(sp.stage_keys[st]) for st in ('front', 'gru', 'regr')]), flush=True)
        gn = lambda n: sum([['tb_g%d' % j, 'lr_g%d' % j] for j in range(n // 2)], []); xn = lambda n: sum([['xtb_g%d' % j, 'xlr_g%d' % j] for j in range(n // 2)], [])
        mods = {'front_ringb%d' % tt: (RingWrapB(CropFrontAff(tc), 2, G['front']).cuda(), tuple(list(base['front_crop%d' % tt][1]) + rf), ['att2', 'aff'] + gn(len(rf)), ['fcs', 'fm', 'net', 'ctx'] + xn(len(rf))),
                'gru_ringb%d' % tt: (RingWrapB(GRUStageRing(tc.raft_stereo, 8, tt, tt), 6, G['gru']).cuda(), tuple(list(base['gru_crop%d' % tt][1]) + rg), ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'] + gn(len(rg)), ['flow', 'mask'] + xn(len(rg))),
                'regr_ringb%d' % S: (RingWrapB(RegrAff(tc), 6, G['regr']).cuda(), tuple(list(base['regr_crop%d' % S][1]) + rr), ['img', 'dep', 'f1', 'f2', 'f3', 'aff'] + gn(len(rr)), ['rot', 'scale', 'opacity', 'res'] + xn(len(rr))),
                'att_rows_l0': base['att_rows_l0'], 'att_rows_l1': base['att_rows_l1']}
        json.dump({'groups': {st: [{'sw': g['sw'], 'r': g['r'], 'C': g['C'], 'n_layers': len(g['layers'])} for g in G[st]] for st in G}}, open(os.path.join(out, 'rings_b.json'), 'w'))
    elif a.ring_mode == 'group':   # grouped rings (lib/tile_mode.ring_groups): one input / output per (store width, ring) group of a stage
        from lib.tile_mode import ring_groups
        G = ring_groups(sp.stage_keys, sp.meta, sp.stores)
        ringsg = lambda st: [z(N, g['C'], k * g['sw'] // w8 + 2 * g['r'], k * g['sw'] // w8 + 2 * g['r']) for g in G[st]]
        rf, rg, rr = ringsg('front'), ringsg('gru'), ringsg('regr')
        print('ring groups: front %d, gru %d, regr %d (layers %d, %d, %d)' % (len(rf), len(rg), len(rr), *[len(sp.stage_keys[st]) for st in ('front', 'gru', 'regr')]), flush=True)
        gn = lambda n: ['ring_g%d' % j for j in range(n)]; xn = lambda n: ['xin_g%d' % j for j in range(n)]
        mods = {'front_ringg%d' % tt: (RingWrapG(CropFrontAff(tc), 2, G['front']).cuda(), tuple(list(base['front_crop%d' % tt][1]) + rf), ['att2', 'aff'] + gn(len(rf)), ['fcs', 'fm', 'net', 'ctx'] + xn(len(rf))),
                'gru_ringg%d' % tt: (RingWrapG(GRUStageRing(tc.raft_stereo, 8, tt, tt), 6, G['gru']).cuda(), tuple(list(base['gru_crop%d' % tt][1]) + rg), ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'] + gn(len(rg)), ['flow', 'mask'] + xn(len(rg))),
                'regr_ringg%d' % S: (RingWrapG(RegrAff(tc), 6, G['regr']).cuda(), tuple(list(base['regr_crop%d' % S][1]) + rr), ['img', 'dep', 'f1', 'f2', 'f3', 'aff'] + gn(len(rr)), ['rot', 'scale', 'opacity', 'res'] + xn(len(rr))),
                'att_rows_l0': base['att_rows_l0'], 'att_rows_l1': base['att_rows_l1']}
        json.dump({'groups': {st: [{'sw': g['sw'], 'r': g['r'], 'C': g['C'], 'n_layers': len(g['layers'])} for g in G[st]] for st in G}}, open(os.path.join(out, 'rings_g.json'), 'w'))
    else:
      rf, rg, rr = rings('front'), rings('gru'), rings('regr')
      print('ring layers: front %d, gru %d, regr %d' % (len(rf), len(rg), len(rr)), flush=True)
      mods = {'front_ring%d' % tt: (RingWrap(CropFrontAff(tc), 2).cuda(), tuple(list(base['front_crop%d' % tt][1]) + rf), ['att2', 'aff'] + ['ring_%d' % i for i in range(len(rf))], ['fcs', 'fm', 'net', 'ctx'] + ['xin_%d' % i for i in range(len(rf))]),
            'gru_ring%d' % tt: (RingWrap(GRUStageRing(tc.raft_stereo, 8, tt, tt), 6).cuda(), tuple(list(base['gru_crop%d' % tt][1]) + rg), ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'] + ['ring_%d' % i for i in range(len(rg))], ['flow', 'mask'] + ['xin_%d' % i for i in range(len(rg))]),
            'regr_ring%d' % S: (RingWrap(RegrAff(tc), 6).cuda(), tuple(list(base['regr_crop%d' % S][1]) + rr), ['img', 'dep', 'f1', 'f2', 'f3', 'aff'] + ['ring_%d' % i for i in range(len(rr))], ['rot', 'scale', 'opacity', 'res'] + ['xin_%d' % i for i in range(len(rr))]),
            'att_rows_l0': base['att_rows_l0'], 'att_rows_l1': base['att_rows_l1']}
      json.dump({'front': len(rf), 'gru': len(rg), 'regr': len(rr)}, open(os.path.join(out, 'rings.json'), 'w'))
if a.dense:
    td = fold(copy.deepcopy(t), frozen=False).eval()   # dense stages: live statistics as the per-frame recipe
    dense_mods = {'front': (Front(td).cuda(), (z(2, 3, 1024, 1024),), ['image'], ['f1', 'f2', 'f3', 'att1', 'att2', 'fcs', 'fm', 'net', 'ctx']),
                  'gru': (DenseGRU(td.raft_stereo, 8).cuda(), (z(2, 192, 128, 128), z(2, 96, 128, 128), z(2, 288, 128, 128), z(2, 1, 128, 128)), ['fm', 'net', 'ctx', 'fi'], ['up']),
                  'regr': (Regr(td).cuda(), (z(2, 3, 1024, 1024), z(2, 1, 1024, 1024), z(2, 32, 512, 512), z(2, 48, 256, 256), z(2, 96, 128, 128)), ['image', 'depth', 'f1', 'f2', 'fcs'], ['rot', 'scale', 'opacity', 'res'])}
    for nm in a.dense: mods[nm] = dense_mods[nm]
with torch.no_grad():
    for name, (mod, args, ins, outs) in mods.items():
        onnx_path, eng = os.path.join(out, name + '.onnx'), os.path.join(out, '%s_%s.engine' % (name, a.precision))
        dyn = name not in ('front', 'gru', 'regr')
        if not os.path.exists(eng):
            torch.onnx.export(mod, args, onnx_path, opset_version=17, input_names=ins, output_names=outs, dynamic_axes={nm: {0: 'N'} for nm in ins + outs} if dyn else None)
            mx = 128 if name.startswith('att_rows') else a.max
            prof = {nm: ((1,) + tuple(x.shape[1:]), (a.opt,) + tuple(x.shape[1:]), (mx,) + tuple(x.shape[1:])) for nm, x in zip(ins, args)} if dyn else None
            t0 = time.time(); build_engine(onnx_path, eng, fp16, prof, a.workspace); print('built %s in %.0f s' % (eng, time.time() - t0), flush=True)
        else:
            print('exists %s' % eng)
print('DONE')
