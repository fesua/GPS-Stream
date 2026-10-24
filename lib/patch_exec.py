import time
import torch
import torch.nn.functional as F
from core.corr import CorrBlock1D
from core.utils.utils import downflow8, coords_grid
from lib.utils import depth2pc
from lib.temporal_gps import TemporalGPS
from lib.temporal_sparse import FrozenNorm, freeze_norms, set_norm_mode
from lib.patch_step3 import PatchStep3

# Step 3, Phase B: the per-frame recipe executed only where it is needed.
#   encoder + epipolar attention: dense (feature encoding = step 1)
#   STSS: dense on the 1/8 map (cheap)
#   context net, cost volume, 8 GRU iterations, convex upsampling: on ROW BANDS of the 1/8 map (full width, the rows of the
#     updated tiles of either view, grown by `halo` cells) - rectified stereo matches along rows, so a band is exact up
#     to the context cut at its top / bottom
#   regressor: on the 2D crop of each view's updated tiles grown by `reg_margin` px; its GroupNorm statistics are taken
#     from the start frame's dense pass when frozen_norm=True (a crop changes the statistics otherwise)
# Outputs as PatchStep3 (dense emulation); differences come only from the context cuts, measured by bench_patch_exec.py.


def row_bands(rows, halo, h):
    """bool (h,) active rows -> list of (r0, r1) bands grown by halo cells, merged when they touch"""
    idx = torch.where(rows)[0].tolist()
    if not idx:
        return []
    bands = []
    start = prev = idx[0]
    for r in idx[1:]:
        if r != prev + 1:
            bands.append((start, prev)); start = r
        prev = r
    bands.append((start, prev))
    out = []
    for r0, r1 in bands:
        r0, r1 = max(r0 - halo, 0), min(r1 + 1 + halo, h)
        if out and r0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], r1))
        else:
            out.append((r0, r1))
    return out


class PatchExec:
    def __init__(self, model: TemporalGPS, patch=64, footprint='tile', margin=0, halo=8, reg_margin=64, frozen_norm=True, key_model=None):
        self.m, self.c, self.footprint, self.margin, self.halo, self.reg_margin, self.frozen = model, patch // 8, footprint, margin, halo, reg_margin, frozen_norm
        self.key = key_model or model
        if frozen_norm:
            freeze_norms(model.teacher.gs_parm_regresser)

    @torch.no_grad()
    def keyframe(self, data):
        reg = self.m.teacher.gs_parm_regresser
        if self.frozen:  # record the dense statistics of THIS (tile) network's regressor on the start frame
            set_norm_mode(reg, 'record'); self.m.keyframe(data); set_norm_mode(reg, 'frozen')
        st = self.key.keyframe(data)
        return {'f3': st['f3'], 'image': st['image'], 'flow': st['flow'], 'g': st['g']}

    @torch.no_grad()
    def step(self, data, state):
        m, t = self.m, self.m.teacher
        rs = t.raft_stereo
        T = {}; tick = lambda: (torch.cuda.synchronize(), time.perf_counter())[1]
        t0 = tick()
        inp = m.inputs(data)
        feat = t.img_encoder(inp['image'])
        c0, c1 = t.loftr_coarse(*feat[2].split(1), None, None)
        feat_cs = torch.cat([c0, c1], 0); f3 = feat[2]
        T['encoder+attention'] = tick() - t0; t0 = tick()
        changed = (torch.sigmoid(m.stss(f3, state['f3'])) > 0.5).float()
        M = PatchStep3.footprint_mask(changed, self.c, self.footprint, self.margin)
        Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
        T['stss'] = tick() - t0; t0 = tick()
        h, w = M.shape[-2:]
        if not M.any():
            T['stereo bands'] = T['regressor crops'] = 0.0
            return {'g': state['g'], 'flow': state['flow'], 'mask': M.float(), 'changed': changed, 'times': T, 'band_rows': 0, 'crop_px': 0}, dict(state)
        # ---- stereo on row bands
        bands = row_bands(M[:, 0].any(0).any(1), self.halo, h)
        flow_new = state['flow'].clone()
        for r0, r1 in bands:
            fcs = feat_cs[:, :, r0:r1]
            *cl, fm1, fm2 = rs.cnet(fcs)
            net = torch.tanh(cl[0][0])
            conv = rs.context_zqr_convs[0]
            ctx = list(conv(torch.relu(cl[0][1])).split(conv.out_channels // 3, 1))
            fmap, fmap_o = torch.cat([fm1, fm2], 0).float(), torch.cat([fm2, fm1], 0).float()
            corr_fn = CorrBlock1D(fmap, fmap_o, radius=rs.args.corr_radius, num_levels=rs.args.corr_levels)
            coords0 = coords_grid(2, r1 - r0, w).to(f3.device)
            fi = downflow8(inp['flow_init'])[:, :, r0:r1]
            coords1 = coords0 + torch.cat([fi, torch.zeros_like(fi)], 1)
            for _ in range(m.iters):
                corr = corr_fn(coords1)
                nl, up_mask, delta = rs.update_module.update_block([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
                net = nl[0]
                delta[:, 1] = 0
                coords1 = coords1 + delta
            flow_new[:, :, 8 * r0:8 * r1] = m.upsample(coords1 - coords0, up_mask)[:, :1]
        flow = torch.where(Mf, flow_new, state['flow'])
        T['stereo bands'] = tick() - t0; t0 = tick()
        # ---- regressor on 2D crops, per view
        depth = m.to_depth(flow, inp)
        gp = state['g']
        g = {'rot': gp['rot'].clone(), 'scale': gp['scale'].clone(), 'opacity': gp['opacity'].clone(), 'depth': gp['depth'].clone()}
        depth_res = depth.clone()
        crop_px = 0
        scales = [inp['image'].shape[-1] // fe.shape[-1] for fe in (feat[0], feat[1], feat_cs)]
        for v in (0, 1):
            if not M[v].any():
                continue
            ys, xs = torch.where(M[v, 0])
            y0, y1 = max(int(ys.min()) * 8 - self.reg_margin, 0), min(int(ys.max()) * 8 + 8 + self.reg_margin, 1024)
            x0, x1 = max(int(xs.min()) * 8 - self.reg_margin, 0), min(int(xs.max()) * 8 + 8 + self.reg_margin, 1024)
            y0, x0 = y0 - y0 % 8, x0 - x0 % 8; y1, x1 = y1 + (-y1) % 8, x1 + (-x1) % 8
            crop_px += (y1 - y0) * (x1 - x0)
            if self.frozen:
                FrozenNorm.view_ids = torch.tensor([v], device=f3.device)
            fe = [fx[v:v + 1, :, y0 // s:y1 // s, x0 // s:x1 // s] for fx, s in zip((feat[0], feat[1], feat_cs), scales)]
            rot, scale, opacity, res = t.gs_parm_regresser(inp['image'][v:v + 1, :, y0:y1, x0:x1], depth[v:v + 1, :, y0:y1, x0:x1], tuple(fe))
            sel = Mf[v:v + 1, :, y0:y1, x0:x1]
            g['rot'][v:v + 1, :, y0:y1, x0:x1] = torch.where(sel, rot, g['rot'][v:v + 1, :, y0:y1, x0:x1])
            g['scale'][v:v + 1, :, y0:y1, x0:x1] = torch.where(sel, scale, g['scale'][v:v + 1, :, y0:y1, x0:x1])
            g['opacity'][v:v + 1, :, y0:y1, x0:x1] = torch.where(sel, opacity, g['opacity'][v:v + 1, :, y0:y1, x0:x1])
            depth_res[v:v + 1, :, y0:y1, x0:x1] = depth[v:v + 1, :, y0:y1, x0:x1] + res
        FrozenNorm.view_ids = None
        g['depth'] = torch.where(Mf, depth_res, gp['depth'])
        g['img'] = torch.where(Mf, inp['image'], gp['img'])
        g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
        g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
        T['regressor crops'] = tick() - t0
        new_state = {'f3': torch.where(M, f3, state['f3']), 'image': torch.where(Mf, inp['image'], state['image']), 'flow': flow, 'g': g}
        return {'g': g, 'flow': flow, 'mask': M.float(), 'changed': changed, 'times': T, 'band_rows': sum(r1 - r0 for r0, r1 in bands), 'crop_px': crop_px}, new_state
