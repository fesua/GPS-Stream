import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.GaussianRender import pts2render
from lib.temporal_gps import TemporalGPS, R, fill_render_data
from lib.patch_step3 import PatchStep3
from train_temporal import to_cuda

# Step 3 (patch-wise estimation of the dynamic regions, GPS-Gaussian+ recipe inside the patches), Phase A: dense
# emulation. Renders views 2 / 3 like test_temporal.py (-> data_process/eval_metrics.py) and prints, for the left
# source view, the disparity difference to the per-frame model outside the patches (2-cell ring, silhouette band) and
# the step across the patch boundaries (ours vs the per-frame model's own step at the same pixels).
# python test_patch3.py --exp patch3_stss_p64 --patch 64 --mask stss [--video cook_spinach 2]
TEACHER = 'experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'
CK = 'experiments/temporal_r8_ref/ckpt/step8000.pth'


def render(data, novel, g, path):
    data = fill_render_data(data, g)
    data['novel_view'] = novel
    with torch.no_grad():
        black = pts2render(data, bg_color=[0, 0, 0])['novel_view']['img_pred'][0].clone()
        white = pts2render(data, bg_color=[1, 1, 1])['novel_view']['img_pred'][0]
    alpha = 1 - (white - black).mean(0)
    cv2.imwrite(path + '.jpg', (black.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(path + '_mask.png', ((alpha > 0.5) * 255).byte().cpu().numpy())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True); p.add_argument('--patch', type=int, default=64); p.add_argument('--mask', default='stss', choices=['stss', 'target']); p.add_argument('--dilate', type=int, default=0, help='cells added around the changed cells before tiling (patch 8 + dilate k = cells + k-cell ring)')
    p.add_argument('--footprint', default='tile', choices=['tile', 'band', 'bbox']); p.add_argument('--margin', type=int, default=0, help='bbox footprint: cells added around each connected changed region')
    p.add_argument('--exec', default='dense', choices=['dense', 'patch', 'sparse', 'trt', 'trt_full', 'sparse_trt', 'perframe'], help='patch: row bands + regressor crops; sparse: tile cells only (lib/patch_sparse.py); trt: per-frame network as a TensorRT engine (lib/patch_trt.py, --engine)'); p.add_argument('--halo', type=int, default=8); p.add_argument('--engine', default='experiments/gps_plus_dynerf_jit2_1007/trt/source_net_f3_it8_fp16.engine'); p.add_argument('--reg_margin', type=int, default=64); p.add_argument('--live_norm', action='store_true'); p.add_argument('--reg_cells', type=int, default=4, help='sparse_trt: regressor crop margin in cells (1/8)'); p.add_argument('--key_self', action='store_true', help='start frame and stored maps from the evaluated network itself (a network trained from the paper weights), not from the frozen original'); p.add_argument('--enc_engine', default='experiments/gps_plus_dynerf_jit2_1007/trt/sparse/enc_fp16.engine'); p.add_argument('--depth_mode', default='sparse', choices=['sparse', 'dense']); p.add_argument('--reg_mode', default='delta', choices=['delta', 'crop', 'dense_pw', 'dense_all'], help='sparse: regressor mode (delta = DeltaCNN); sparse_trt: crop or dense_all'); p.add_argument('--tile_ctx', action='store_true', help='sparse: stored-context decoding (variant D)'); p.add_argument('--graph', action='store_true', help='sparse_trt: crop / ring engines launched as CUDA graphs'); p.add_argument('--dump_depth', action='store_true', help='also save the inverse depth of both source views (the values the Gaussians are placed with) per frame: experiments/<exp>/depth_<scene>_process/<name>.npz')
    p.add_argument('--ckpt', default=CK); p.add_argument('--scenes', nargs='+', default=['cook_spinach', 'cut_roasted_beef', 'flame_steak', 'sear_steak'])
    p.add_argument('--sets', type=int, nargs='+', default=[1, 2, 3]); p.add_argument('--frames', type=int, nargs=2, default=[220, 300])
    p.add_argument('--video', nargs=2, default=None, metavar=('SCENE', 'SET'), help='also write the overlay video of this scene / set (left view)')
    p.add_argument('--teacher', default=TEACHER, help='per-frame network weights (e.g. fine-tuned inside the patch pipeline by train_patch3.py)')
    a = p.parse_args()
    cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.dataset.inverse_depth_init = 0.5; cfg.raft.val_iters = cfg.raft.train_iters = 8
    teacher = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher.load_state_dict(torch.load(a.teacher, map_location='cuda')['network'])
    ck = torch.load(a.ckpt, map_location='cuda')
    model = TemporalGPS(teacher, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1), full_attention=ck.get('full_attention', False), temporal_attention=ck.get('temporal_attention')).cuda()
    model.load_state_dict(ck['model'], strict=False)
    key_model = None
    if a.teacher != TEACHER and not a.key_self:  # tiles from the fine-tuned per-frame network, the start frame from the frozen original one
        teacher.load_state_dict(torch.load(a.teacher, map_location='cuda')['network'])
        teacher0 = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network'])
        key_model = TemporalGPS(teacher0, iters=8, radius=ck.get('radius', R), update_dilation=ck.get('update_dilation', 1)).cuda(); key_model.load_state_dict(ck['model'], strict=False)
        teacher0.load_state_dict(torch.load(TEACHER, map_location='cuda')['network']); key_model.eval()
    model.eval()
    runner = PatchStep3(model, a.patch, a.mask, a.dilate, a.footprint, a.margin, key_model)
    if a.exec == 'patch':
        from lib.patch_exec import PatchExec
        runner = PatchExec(model, a.patch, a.footprint, a.margin, a.halo, a.reg_margin, not a.live_norm, key_model)
    elif a.exec == 'sparse':
        from lib.patch_sparse import PatchSparse
        runner = PatchSparse(model, a.patch, a.halo, key_model=key_model, dense_reg=a.reg_mode != 'delta', reg_crop=a.reg_cells if a.reg_mode == 'crop' else None, depth_sparse=a.depth_mode == 'sparse', reg_mode=a.reg_mode, tile_ctx=a.tile_ctx)
    elif a.exec == 'sparse_trt':   # tile-cell execution with crop engines (lib/patch_sparse_trt.py); --engine = engine directory
        from lib.patch_sparse_trt import PatchSparseTRT
        runner = PatchSparseTRT(model, a.engine, a.patch, a.halo, a.reg_cells, key_model=key_model, enc_engine=a.enc_engine, depth_sparse=a.depth_mode == 'sparse', reg_mode='dense_all' if a.reg_mode == 'dense_all' else 'crop', tile_ctx=a.tile_ctx, graph=a.graph)
    elif a.exec == 'perframe':   # the per-frame network on every frame (no tiles): e.g. the paper weights without DyNeRF adaptation
        class PerFrame:
            def keyframe(self, d):
                st = model.keyframe(d); return {'f3': st['f3'], 'image': st['image'], 'flow': st['flow'], 'g': st['g']}
            def step(self, d, state):
                st = model.keyframe(d); M = torch.ones_like(st['f3'][:, :1])
                return {'g': st['g'], 'flow': st['flow'], 'mask': M, 'changed': M, 'teach_flow': st['flow'], 'teach_g': st['g']}, {'f3': st['f3'], 'image': st['image'], 'flow': st['flow'], 'g': st['g']}
        runner = PerFrame()
    elif a.exec in ('trt', 'trt_full'):
        from lib.patch_trt import PatchTRT
        runner = PatchTRT(model, a.engine, a.patch, full=a.exec == 'trt_full')   # trt_full: the per-frame model as an engine, no tiles
    acc = {k: [0.0, 0, 0] for k in ('ring', 'band', 'outside')}; seam = [0.0, 0.0, 0.0, 0]; stats = []
    vid = None
    for scene in a.scenes:
        ds = StereoHumanDataset(cfg.dataset, phase=scene + '_process'); idx = {n: i for i, n in enumerate(ds.sample_list)}
        out_dir = 'experiments/%s/show_%s_process' % (a.exp, scene); Path(out_dir).mkdir(parents=True, exist_ok=True)
        for s in a.sets:
            make_video = a.video is not None and a.video[0] == scene and int(a.video[1]) == s
            if make_video:
                vpath = '/home/imlab/js/GPS_plus_new/experiments/videos/patch3_%s_%s_s%d.mp4' % (a.exp, scene, s)
                vid = cv2.VideoWriter(vpath + '.tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1536, 512 + 36))
            names = ['%s_s%d_%04d' % (scene, s, t) for t in range(*a.frames)]
            for t, name in enumerate(tqdm(names, desc='%s s%d' % (scene, s))):
                d2 = to_cuda(torch.utils.data.default_collate([ds.get_item(idx[name], novel_id=[2])]))
                n3 = to_cuda(torch.utils.data.default_collate([ds.get_item(idx[name], novel_id=[3])]))['novel_view']
                with torch.no_grad():
                    if t == 0:
                        state = runner.keyframe(d2); g = state['g']; rec = {}
                    else:
                        out, state = runner.step(d2, state); g = out['g']
                        if 'teach_flow' not in out:
                            out['teach_flow'] = model.keyframe(d2)['flow']
                        if out.get('flow') is None:
                            out['flow'] = out['teach_flow']  # trt: no disparity map kept; band numbers are not meaningful for it
                        M = out['mask'][0, 0] > 0.5; Mf = F.interpolate(out['mask'][:1], scale_factor=8, mode='nearest')[0, 0] > 0.5
                        pf, ours = out['teach_flow'][0, 0], out['flow'][0, 0]
                        err = (ours - pf).abs()
                        ring_c = (F.max_pool2d(out['mask'][:1], 5, 1, 2)[0, 0] > 0.5) & ~M
                        near = F.interpolate((F.max_pool2d(out['mask'][:1], 7, 1, 3) > 0.5).float(), scale_factor=8, mode='nearest')[0, 0] > 0.5
                        edge = (F.max_pool2d(pf[None, None], 9, 1, 4) + F.max_pool2d(-pf[None, None], 9, 1, 4))[0, 0] > 15
                        band = (F.max_pool2d(edge[None, None].float(), 25, 1, 12)[0, 0] > 0.5) & near & ~Mf
                        ring = F.interpolate(ring_c[None, None].float(), scale_factor=8, mode='nearest')[0, 0] > 0.5
                        for k, msk in (('ring', ring), ('band', band), ('outside', ~Mf)):
                            if msk.any():
                                acc[k][0] += err[msk].sum().item(); acc[k][1] += msk.sum().item(); acc[k][2] += (err[msk] > 8).sum().item()
                        # steps across the patch boundary (horizontal + vertical neighbour pairs with one pixel inside, one outside)
                        for dx, dy in ((1, 0), (0, 1)):
                            a1 = Mf[dy:, dx:] != Mf[:1024 - dy, :1024 - dx]
                            d_ours = (ours[dy:, dx:] - ours[:1024 - dy, :1024 - dx])[a1]; d_pf = (pf[dy:, dx:] - pf[:1024 - dy, :1024 - dx])[a1]
                            seam[0] += d_ours.abs().sum().item(); seam[1] += d_pf.abs().sum().item(); seam[2] += (d_ours - d_pf).abs().sum().item(); seam[3] += a1.sum().item()
                        rec = {'mask': out['mask'].mean().item(), 'changed': out['changed'].mean().item(), 'band_err': err[band].mean().item() if band.any() else 0.0}
                        if make_video:
                            img = ((d2['lmain']['img'][0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 0.5 + 0.5) * 255).astype(np.uint8)
                            ov = img.copy(); stable = ((pf - prev_pf).abs() < 3).cpu().numpy()
                            f8 = ((err > 8).cpu().numpy() & stable); f20 = ((err > 20).cpu().numpy() & stable)
                            ov[f8] = (0.35 * ov[f8] + 0.65 * np.array([0, 0, 255])).astype(np.uint8); ov[f20] = (255, 0, 255)
                            for cells, col, th in ((out['changed'][0, 0] > 0.5, (0, 255, 255), 3), (M, (255, 255, 255), 5)):
                                mk = cv2.resize((cells.cpu().numpy() * 255).astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
                                ov[cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((th, th), np.uint8)) > 0] = col
                            lo, hi = np.percentile(pf.cpu().numpy(), 1), np.percentile(pf.cpu().numpy(), 99)
                            cm = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
                            col_ = lambda x: cv2.applyColorMap((np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cm)
                            dmap = col_(ours.cpu().numpy()); pmap = col_(pf.cpu().numpy())
                            mk = cv2.resize((M.cpu().numpy() * 255).astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
                            dmap[cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0] = (255, 255, 255)
                            bar = np.full((36, 1536, 3), 24, np.uint8)
                            cv2.putText(bar, '%s | patch %d px (dilate %d), mask %s | yellow STSS cells, white patch border, red |ours-perframe|>8px, magenta >20px (where per-frame stable) | right: our disparity (patch borders) | per-frame disparity' % (name, a.patch, a.dilate, a.mask), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                            vid.write(np.concatenate([bar, np.concatenate([cv2.resize(ov, (512, 512), interpolation=cv2.INTER_AREA), cv2.resize(dmap, (512, 512), interpolation=cv2.INTER_AREA), cv2.resize(pmap, (512, 512), interpolation=cv2.INTER_AREA)], 1)], 0))
                    prev_pf = (out['teach_flow'] if t > 0 else (state['flow'] if state.get('flow') is not None else model.keyframe(d2)['flow']))[0, 0].clone()  # trt: no disparity map in the state
                if a.dump_depth:   # inverse depth (depth2pc places a pixel at z = 1 / value) of the left / right source views, and the valid mask
                    dd = 'experiments/%s/depth_%s_process' % (a.exp, scene); Path(dd).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed('%s/%s.npz' % (dd, name), inv_depth=g['depth'][:, 0].half().cpu().numpy(), valid=g['valid'].view(g['depth'].shape[0], *g['depth'].shape[-2:]).cpu().numpy())
                render(d2, d2['novel_view'], g, '%s/%s_02' % (out_dir, name))
                render(d2, n3, g, '%s/%s_03' % (out_dir, name))
                stats.append({'scene': scene, 'set': 's%d' % s, 'frame': int(name[-4:]), **rec})
            if make_video:
                vid.release()
                os.system('/home/imlab/anaconda3/envs/gps_plus/bin/ffmpeg -y -loglevel error -i %s.tmp.mp4 -c:v libx264 -pix_fmt yuv420p -crf 18 %s && rm %s.tmp.mp4' % (vpath, vpath, vpath))
                print('video', vpath)
    json.dump(stats, open('experiments/%s/stats.json' % a.exp, 'w'))
    st = [x for x in stats if 'mask' in x]
    f = lambda k: (acc[k][0] / max(acc[k][1], 1), 100 * acc[k][2] / max(acc[k][1], 1))
    print('%s: footprint %s, patch %d px, dilate %d, margin %d, mask %s | frames %d | updated area %.2f%% (changed cells %.2f%%) | disparity vs per-frame (left view, px, >8 px): 2-cell ring %.2f (%.1f%%), silhouette band outside the patches %.2f (%.1f%%), all outside %.2f (%.1f%%) | patch-boundary step: ours %.2f px, per-frame at the same pixels %.2f, excess %.2f (%d pairs)' % (
        a.exp, a.footprint, a.patch, a.dilate, a.margin, a.mask, len(st), 100 * np.mean([x['mask'] for x in st]), 100 * np.mean([x['changed'] for x in st]), *f('ring'), *f('band'), *f('outside'), seam[0] / max(seam[3], 1), seam[1] / max(seam[3], 1), seam[2] / max(seam[3], 1), seam[3]))
