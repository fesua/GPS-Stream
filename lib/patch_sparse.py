import time
import torch
import torch.nn.functional as F
from core.corr import CorrBlock1D
from core.utils.utils import downflow8, coords_grid
from lib.utils import depth2pc
from lib.temporal_gps import TemporalGPS
from lib.temporal_sparse import FrozenNorm, freeze_norms, set_norm_mode, crop_tiles
from lib.temporal_patch import put_tiles, patch_tiles
from lib.patch_step3 import PatchStep3
from lib.frozen_norm import freeze_group_norms, set_norm_mode as set_gn_mode
try:   # DeltaCNN is needed only for reg_mode 'delta' (PyTorch regressor on the changed pixels); the crop / TensorRT paths do not use it
    from lib.deltacnn_model import DCRegressor, fold_norm, prepare, cl
except ImportError:
    DCRegressor = fold_norm = prepare = cl = None
from lib import tile_mode
from lib.tile_mode import TileCtx, convex_upsample_ring

# Step 3 executed on the tile cells only (the per-frame recipe, no re-computation outside the tiles):
#   encoder: dense (step 1). STSS: dense on the 1/8 map.
#   epipolar attention (LoFTR, per-row): QUERIES = the tile cells only, keys / values = the other view's rows (the
#     current encoder features for the first layer, the stored layer-1 outputs elsewhere for the second) -> cost
#     proportional to the number of tile cells; cells outside the tiles keep the stored outputs.
#   attention residual block, context net, GRU x 8, convex upsampling: on a batch of tile CROPS (tile + halo cells,
#     one crop per tile, all crops through each layer as one batch); the crop's neighbours come from the stored maps
#     (patchwork), zeros outside the image; only the tile core is written back. The GRU starts fresh (tanh(context),
#     constant inverse-depth start) as in the per-frame recipe; the cost volume of a crop is the full row of the
#     other view's feature map (current in its tiles, stored elsewhere), so matching is unrestricted.
#   regressor: DeltaCNN (changed pixels only, exact), GroupNorm statistics of the start frame.
# Normalisation layers of the crop stages use the statistics recorded on the start frame (FrozenNorm, per view).
# The halo bounds the context a tile sees per stage; the GRU's 8 iterations would need ~50 cells for exactness, so the
# tile results are an approximation whose size bench_patch_sparse.py measures against the dense emulation.


def head_activations(reg, raw):
    rot = F.normalize(raw[:, :4], dim=1)
    scale = torch.clamp_max(reg.scale_head[-1](raw[:, 4:7]), 0.002)
    opacity = reg.opacity_head[-1](raw[:, 7:8])
    res = reg.depth_head[-1](raw[:, 8:9]) * 0.5
    return rot, scale, opacity, res


def att_sparse(layer, x, src, cells):
    """Crossatt_EncoderLayer on the query cells only. x, src (1, C, H, W); cells (H, W) bool -> (N, C) new x at the cells
    (same arithmetic as the dense layer: per-row attention of each query over the source row)"""
    C, H, W = x.shape[1:]
    ys, xs = torch.where(cells)
    rows = torch.unique(ys)
    xq = x[0].permute(1, 2, 0)                                  # H W C
    q_all = layer.q_proj(layer.norm1(xq))                       # H W C
    s_all = layer.norm1(src[0].permute(1, 2, 0)); k_all, v_all = layer.k_proj(s_all), layer.v_proj(s_all)
    # pack the query cells by row (padded to the longest row)
    counts = torch.bincount(ys, minlength=H)[rows]
    n_max = int(counts.max())
    order = torch.argsort(ys * W + xs)
    ys_o, xs_o = ys[order], xs[order]
    row_pos = torch.searchsorted(rows, ys_o)
    first = torch.cumsum(F.pad(counts, (1, 0)), 0)[:-1]
    slot = torch.arange(len(ys_o), device=x.device) - first[row_pos]
    Q = torch.zeros(len(rows), n_max, C, device=x.device, dtype=q_all.dtype)
    Q[row_pos, slot] = q_all[ys_o, xs_o]
    K, V = k_all[rows], v_all[rows]                              # R W C
    m, _ = layer.attention(Q, K, V)                             # R n_max C
    m = layer.merge(m[row_pos, slot])                           # N C (in sorted order)
    xo = xq[ys_o, xs_o]
    out = xo + layer.norm2(layer.mlp(torch.cat([xo, m], -1)))
    return out, ys_o, xs_o


class PatchSparse:
    def __init__(self, model: TemporalGPS, patch=64, halo=4, key_model=None, dense_reg=False, reg_crop=None, depth_sparse=True, reg_mode=None, tile_ctx=False):
        self.m, self.k, self.halo = model, patch // 8, halo
        self.tile_ctx = tile_ctx    # variant D: every conv of the tile stages reads its ring from a per-layer full-frame store (lib/tile_mode.py); use with halo 0
        self.stores = None
        if tile_ctx:
            tile_mode.install()
        self.key = key_model or model   # the start frame (and the stored maps outside the tiles) come from this network: the frozen original one when `model` is fine-tuned
        # variants (each trained under its own condition):
        #   depth_sparse: attention / residual block / cnet / GRU on the tile crops only (True) or the dense per-frame recipe every frame (False)
        #   reg_mode: 'delta' = DeltaCNN on the changed pixels (PyTorch inference); 'dense_pw' = dense module on the patchwork, tiles written back;
        #             'crop' = tile crops with `reg_crop` margin cells (the TensorRT crop regressor); 'dense_all' = dense module on the patchwork, output used everywhere
        if reg_mode is None:
            reg_mode = 'crop' if (dense_reg and reg_crop is not None) else ('dense_pw' if dense_reg else 'delta')
        self.depth_sparse, self.reg_mode, self.reg_crop = depth_sparse, reg_mode, (reg_crop or 0)
        if reg_mode == 'delta' and DCRegressor is None:
            raise ImportError("reg_mode 'delta' needs the deltacnn package (pip install from https://github.com/facebookresearch/DeltaCNN); use reg_mode 'crop' otherwise")
        self.dense_reg = reg_mode != 'delta'
        t = model.teacher
        self.rs = t.raft_stereo
        freeze_norms(t.loftr_coarse.resid); freeze_norms(self.rs.cnet)
        self.gn = freeze_group_norms(t.gs_parm_regresser)
        self.dc = None
        self.crop_norm = 'frozen'  # statistics used by the crop passes ('live' only for the implementation check)
        self.mask_override = None  # training: (B,1,h,w) changed-cell map to use instead of the STSS prediction
        self.debug = None  # dict -> intermediates of the last step are stored (bench --check)

    def _modes(self, mode):
        t = self.m.teacher
        set_norm_mode(t.loftr_coarse.resid, mode); set_norm_mode(self.rs.cnet, mode); set_gn_mode(t.gs_parm_regresser, mode)

    def _dense(self, m, data):
        """the dense per-frame recipe with the modules of `m` (= self.m or self.key): the stored maps of the start frame"""
        t, rs = m.teacher, m.teacher.raft_stereo
        inp = m.inputs(data)
        feat = t.img_encoder(inp['image']); f3 = feat[2]
        record = self.tile_ctx and m is self.m
        if record:
            self.stores = {}; TileCtx.begin('record', self.stores)
        try:
            out = self._dense_rest(m, inp, feat)
        finally:
            if record:
                self.meta, order, marks = TileCtx.meta, TileCtx.order, TileCtx.marks
                names = ['front', 'gru', 'regr', 'end']
                self.stage_keys = {a: order[marks[a]:marks[b]] for a, b in zip(names[:-1], names[1:])}
                TileCtx.end()
        return out

    def _dense_rest(self, m, inp, feat):
        t, rs = m.teacher, m.teacher.raft_stereo
        f3 = feat[2]
        lo = t.loftr_coarse; L = lo.layers
        a0, a1 = f3[:1], f3[1:]
        a0 = L[0](a0, a1); a1 = L[0](a1, a0); att1 = torch.cat([a0, a1])
        a0 = L[1](a0, a1); a1 = L[1](a1, a0); att2 = torch.cat([a0, a1])
        TileCtx.mark('front')
        fcs = lo.resid(att2)
        *cl_, fm1, fm2 = rs.cnet(fcs)
        net = torch.tanh(cl_[0][0]); conv = rs.context_zqr_convs[0]
        ctx = list(conv(torch.relu(cl_[0][1])).split(conv.out_channels // 3, 1))
        fmap, fmap_o = torch.cat([fm1, fm2], 0).float(), torch.cat([fm2, fm1], 0).float()
        corr_fn = CorrBlock1D(fmap, fmap_o, radius=rs.args.corr_radius, num_levels=rs.args.corr_levels)
        h, w = f3.shape[-2:]; coords0 = coords_grid(2, h, w).to(f3.device)
        TileCtx.mark('gru')
        fi = downflow8(inp['flow_init']); coords1 = coords0 + torch.cat([fi, torch.zeros_like(fi)], 1)
        upd = rs.update_module.update_block
        for _ in range(m.iters):
            corr = corr_fn(coords1)
            nl, up_mask, delta = upd([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
            net = nl[0]; delta[:, 1] = 0; coords1 = coords1 + delta
        if TileCtx.active and TileCtx.mode == 'record':
            self.stores['flow8'] = (coords1 - coords0).detach().clone()
        flow = m.upsample(coords1 - coords0, up_mask)[:, :1]
        depth = m.to_depth(flow, inp)
        TileCtx.mark('regr')
        g = m.gaussians(t.gs_parm_regresser, inp, depth, (feat[0], feat[1], fcs))   # dense regressor: records its statistics
        TileCtx.mark('end')
        self.coords0 = coords0
        return {'f3': f3, 'att1': att1, 'att2': att2, 'feat_cs': fcs, 'fmap': fmap, 'ctx': ctx, 'flow': flow, 'g': g, 'image': inp['image'],
                'reg_in': (inp['image'], depth, (feat[0], feat[1], fcs))}

    @torch.no_grad()
    def keyframe(self, data):
        t = self.m.teacher
        self._modes('record')
        st = self._dense(self.m, data)          # records the normalisation statistics of `model` (start frame, per view)
        self._modes('live')                     # dense passes (start frame, the dense emulation used as reference) keep live statistics
        if self.key is not self.m:
            st = self._dense(self.key, data)    # stored maps from the frozen original network
        inp = self.m.inputs(data); depth = st['reg_in'][1]; feats = list(st['reg_in'][2])
        reg = t.gs_parm_regresser
        if self.reg_mode != 'delta':
            return st
        if self.dc is None:
            self.dc = [prepare(DCRegressor(reg, 0.0, dense_feat=True, sample=v)) for v in range(2)]
        else:
            from deltacnn.sparse_layers import DCBatchNorm2d
            for mod in self.dc:
                mod.reset_layers()
                for d in [x for x in mod.modules() if isinstance(x, DCBatchNorm2d)]:
                    fold_norm(d, d.source, d.sample); d.first_iter = True
        for v in range(2):
            self.dc[v](cl(inp['image'][v:v + 1]), cl(depth[v:v + 1]), [cl(x[v:v + 1]) for x in feats])
        return st

    def _pads(self, h8):
        """padding in cells so that the tile size divides the 1/8 map (0 for 128 px tiles on a 128-cell map; 16 for 192 px)"""
        k = self.k
        return (-h8) % k

    def _p8(self, x, pad):
        return F.pad(x, (0, pad, 0, pad)) if pad else x

    def _ppx(self, x, pad, s=8):
        return F.pad(x, (0, pad * s, 0, pad * s)) if pad else x

    def _u(self, x, h, w):
        return x[..., :h, :w].contiguous() if (x.shape[-2] != h or x.shape[-1] != w) else x   # contiguous: the renderer views the maps

    def _dense_flow(self, inp, feat):
        """the dense per-frame stereo recipe on the current frame (attention, residual block, cnet, GRU x iters, upsampling); live statistics"""
        m, t, rs = self.m, self.m.teacher, self.rs
        f3 = feat[2]; L = t.loftr_coarse.layers
        a0, a1 = f3[:1], f3[1:]
        a0 = L[0](a0, a1); a1 = L[0](a1, a0); att1 = torch.cat([a0, a1])
        a0 = L[1](a0, a1); a1 = L[1](a1, a0); att2 = torch.cat([a0, a1])
        fcs = t.loftr_coarse.resid(att2)
        *cl_, fm1, fm2 = rs.cnet(fcs)
        net = torch.tanh(cl_[0][0]); conv = rs.context_zqr_convs[0]
        ctx = list(conv(torch.relu(cl_[0][1])).split(conv.out_channels // 3, 1))
        fmap, fmap_o = torch.cat([fm1, fm2], 0).float(), torch.cat([fm2, fm1], 0).float()
        corr_fn = CorrBlock1D(fmap, fmap_o, radius=rs.args.corr_radius, num_levels=rs.args.corr_levels)
        h, w = f3.shape[-2:]; coords0 = coords_grid(2, h, w).to(f3.device)
        fi = downflow8(inp['flow_init']); coords1 = coords0 + torch.cat([fi, torch.zeros_like(fi)], 1)
        upd = rs.update_module.update_block
        for _ in range(m.iters):
            corr = corr_fn(coords1)
            nl, up_mask, delta = upd([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
            net = nl[0]; delta[:, 1] = 0; coords1 = coords1 + delta
        flow = m.upsample(coords1 - coords0, up_mask)[:, :1]
        return att1, att2, fcs, fmap, ctx, flow

    def _step_dense_depth(self, data, state, inp, feat, f3, changed, M, Mf, Mp, idx, T, tick, t0):
        """variant B: dense depth every frame (live statistics), regressor on the tiles only (reg_mode crop / dense_pw / delta), tiles written back"""
        m, t = self.m, self.m.teacher
        k, halo = self.k, self.halo
        h8, w8 = f3.shape[-2:]; H, W = inp['image'].shape[-2:]
        pad = self._pads(h8); P8 = lambda x: self._p8(x, pad); Ppx = lambda x, s=8: self._ppx(x, pad, s)
        self._modes('live')
        att1, att2, fcs, fmap, ctx, flow = self._dense_flow(inp, feat)
        T['attention'] = tick() - t0; t0 = tick(); T['cnet'] = 0.0
        T['gru'] = 0.0
        depth = m.to_depth(flow, inp)
        tt = k + 2 * halo
        img0, dep0, f0 = state['reg_in']
        img_pw, dep_pw, feats_pw = inp['image'], depth, [feat[0], feat[1], fcs]   # the depth inputs are dense (new everywhere)
        new_state_base = lambda g: {'f3': torch.where(M, f3, state['f3']), 'att1': att1, 'att2': att2, 'feat_cs': fcs, 'fmap': fmap, 'ctx': ctx, 'flow': flow, 'g': g,
                                    'image': inp['image'], 'reg_in': (img_pw, dep_pw, feats_pw)}
        if idx.shape[0] == 0:
            T['regressor'] = 0.0
            return {'g': state['g'], 'flow': flow, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': 0}, new_state_base(state['g'])
        Mc = crop_tiles(Mp, idx, k, 0); Mcf = F.interpolate(Mc.float(), scale_factor=8, mode='nearest') > 0.5
        gp = state['g']
        if self.reg_mode == 'crop':
            mr = self.reg_crop
            cr = lambda x, s: crop_tiles(Ppx(x, s), idx, k * s, mr * s); cm = lambda x: x[..., 8 * mr:8 * (mr + k), 8 * mr:8 * (mr + k)]
            set_gn_mode(t.gs_parm_regresser, self.crop_norm, sample_ids=idx[:, 0])
            rot_p, scale_p, opac_p, res_p = t.gs_parm_regresser(cr(img_pw, 8), cr(dep_pw, 8), (cr(feats_pw[0], 4), cr(feats_pw[1], 2), cr(feats_pw[2], 1)))
            set_gn_mode(t.gs_parm_regresser, 'live', sample_ids=None)
            g = {kk: Ppx(gp[kk]).clone() for kk in ('rot', 'scale', 'opacity', 'depth')}
            put_tiles(g['rot'], cm(rot_p), idx, Mcf); put_tiles(g['scale'], cm(scale_p), idx, Mcf); put_tiles(g['opacity'], cm(opac_p), idx, Mcf)
            put_tiles(g['depth'], crop_tiles(Ppx(depth), idx, 8 * k, 0) + cm(res_p), idx, Mcf)
            g = {kk: self._u(v, H, W) for kk, v in g.items()}
        else:
            if self.reg_mode == 'dense_pw':
                set_gn_mode(t.gs_parm_regresser, self.crop_norm)
                rot, scale, opacity, res = t.gs_parm_regresser(img_pw, dep_pw, feats_pw)
                set_gn_mode(t.gs_parm_regresser, 'live')
            else:
                raw = torch.cat([self.dc[v](cl(img_pw[v:v + 1]), cl(dep_pw[v:v + 1]), [cl(x[v:v + 1]) for x in feats_pw]) for v in range(2)], 0)
                rot, scale, opacity, res = head_activations(t.gs_parm_regresser, raw)
            g = {'rot': torch.where(Mf, rot, gp['rot']), 'scale': torch.where(Mf, scale, gp['scale']), 'opacity': torch.where(Mf, opacity, gp['opacity']), 'depth': torch.where(Mf, depth + res, gp['depth'])}
        g['img'] = torch.where(Mf, inp['image'], gp['img']); g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
        g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
        T['regressor'] = tick() - t0
        return {'g': g, 'flow': flow, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state_base(g)

    @torch.no_grad()
    def step(self, data, state):
        m, t, rs = self.m, self.m.teacher, self.rs
        k, halo = self.k, self.halo
        T = {}; tick = lambda: (torch.cuda.synchronize(), time.perf_counter())[1]
        t0 = tick()
        inp = m.inputs(data)
        feat = t.img_encoder(inp['image']); f3 = feat[2]
        h8, w8 = f3.shape[-2:]; H, W = inp['image'].shape[-2:]
        pad = self._pads(h8); P8 = lambda x: self._p8(x, pad); Ppx = lambda x, s=8: self._ppx(x, pad, s); U8 = lambda x: self._u(x, h8, w8); Upx = lambda x: self._u(x, H, W)
        T['encoder'] = tick() - t0; t0 = tick()
        stss_logits = m.stss(f3, state['f3'])
        changed = self.mask_override if self.mask_override is not None else (torch.sigmoid(stss_logits) > 0.5).float()
        M = PatchStep3.footprint_mask(changed, k, 'tile', 0)                 # on the unpadded map (edge tiles clipped)
        Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
        Mp = P8(M)                                                            # padded map: tiles are whole
        idx = patch_tiles(Mp, k)
        T['stss'] = tick() - t0; t0 = tick()
        if idx.shape[0] == 0:
            for key in ('attention', 'cnet', 'gru', 'regressor'): T[key] = 0.0
            return {'g': state['g'], 'flow': state['flow'], 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': 0}, dict(state)
        if not self.depth_sparse:   # variant B: depth by the dense per-frame recipe every frame; only the regressor is tile-sparse
            return self._step_dense_depth(data, state, inp, feat, f3, changed, M, Mf, Mp, idx, T, tick, t0)
        # ---- epipolar attention at the tile cells (per-view copies, each written once before it is read)
        L = t.loftr_coarse.layers
        f3p = P8(f3)
        a1 = [P8(state['att1'][v:v + 1]).clone() for v in (0, 1)]
        for v in (0, 1):
            if M[v, 0].any():
                src = f3p[1 - v:2 - v] if v == 0 else a1[0]
                out, ys, xs = att_sparse(L[0], f3p[v:v + 1], src, Mp[v, 0]); a1[v][0, :, ys, xs] = out.t()
        a2 = [P8(state['att2'][v:v + 1]).clone() for v in (0, 1)]
        for v in (0, 1):
            if M[v, 0].any():
                src = a1[1] if v == 0 else a2[0]
                out, ys, xs = att_sparse(L[1], a1[v], src, Mp[v, 0]); a2[v][0, :, ys, xs] = out.t()
        att1, att2 = torch.cat(a1), torch.cat(a2)
        crop = lambda x, s=1: crop_tiles(x, idx, k * s, halo * s)
        core = lambda x, s=1: x[..., halo * s:(halo + k) * s, halo * s:(halo + k) * s]
        Mc = crop(Mp)[..., halo:halo + k, halo:halo + k]
        Mcf = F.interpolate(Mc.float(), scale_factor=8, mode='nearest') > 0.5
        FrozenNorm.view_ids = idx[:, 0]
        self._modes(self.crop_norm)   # crop passes: statistics of the start frame, per view
        if self.tile_ctx:
            TileCtx.begin('tile', self.stores, idx, k, w8 + pad)
        fcs_p = t.loftr_coarse.resid(crop(att2))
        feat_cs = P8(state['feat_cs']).clone(); put_tiles(feat_cs, core(fcs_p), idx, Mc)
        T['attention'] = tick() - t0; t0 = tick()
        # ---- context net on the crops
        cnet = rs.cnet
        fm_p = cnet.conv2(fcs_p).float()
        hid_p = cnet.outputs08[0](fcs_p)
        conv = rs.context_zqr_convs[0]
        ctx_p = list(conv(torch.relu(cnet.outputs08[1](fcs_p))).split(conv.out_channels // 3, 1))
        fmap = P8(state['fmap']).clone(); put_tiles(fmap, core(fm_p), idx, Mc)
        ctx = [P8(c).clone() for c in state['ctx']]
        for c, cp in zip(ctx, ctx_p): put_tiles(c, core(cp), idx, Mc)
        T['cnet'] = tick() - t0; t0 = tick()
        # ---- cost volume of each crop against the other view's full rows, GRU, upsampling
        tt = k + 2 * halo
        fo = F.pad(fmap, (0, 0, halo, halo))                                        # rows outside the image = 0, as the crops
        ar = torch.arange(tt, device=f3.device)
        Y = (idx[:, 1] * k)[:, None] + ar[None, :]
        rows_other = fo[(1 - idx[:, 0])[:, None], :, Y].permute(0, 2, 1, 3).contiguous()   # (n, C, tt, Wp)
        corr_fn = CorrBlock1D(fm_p, rows_other, radius=rs.args.corr_radius, num_levels=rs.args.corr_levels)
        if self.coords0.shape[-1] != w8 + pad:
            self.coords0 = coords_grid(2, h8 + pad, w8 + pad).to(f3.device)
        c0_p = crop(self.coords0)
        fi_p = crop(downflow8(Ppx(inp['flow_init'])))
        c1_p = c0_p + torch.cat([fi_p, torch.zeros_like(fi_p)], 1)
        net_p = torch.tanh(hid_p)
        upd = rs.update_module.update_block
        for _ in range(m.iters):
            corr = corr_fn(c1_p)
            nl, up_mask, delta = upd([net_p], [ctx_p], corr, c1_p - c0_p, iter32=False, iter16=False)
            net_p = nl[0]; delta[:, 1] = 0; c1_p = c1_p + delta
        up_p = (convex_upsample_ring(c1_p - c0_p, up_mask, self.stores['flow8']) if self.tile_ctx else m.upsample(c1_p - c0_p, up_mask))[:, :1]
        if self.debug is not None:
            self.debug.update(att1=att1, att2=att2, feat_cs=feat_cs, fmap=fmap, hid=hid_p, ctx0=ctx_p[0], c1=c1_p, up=up_p, idx=idx)
        flow = Ppx(state['flow']).clone(); put_tiles(flow, core(up_p, 8), idx, Mcf)
        FrozenNorm.view_ids = None
        self._modes('live')
        T['gru'] = tick() - t0; t0 = tick()
        # ---- regressor on the patchwork inputs (unpadded maps)
        flow_u = Upx(flow)
        depth = m.to_depth(flow_u, inp)
        img0, dep0, f0 = state['reg_in']
        img_pw = torch.where(Mf, inp['image'], img0); dep_pw = torch.where(Mf, depth, dep0)
        up_s = lambda s: F.interpolate(M.float(), scale_factor=8 // s, mode='nearest') > 0.5
        feats_pw = [torch.where(up_s(inp['image'].shape[-1] // x.shape[-1]), x, r) for x, r in zip((feat[0], feat[1], U8(feat_cs)), f0)]
        new_state_base = lambda g: {'f3': torch.where(M, f3, state['f3']), 'att1': U8(att1), 'att2': U8(att2), 'feat_cs': U8(feat_cs), 'fmap': U8(fmap), 'ctx': [U8(c) for c in ctx], 'flow': flow_u, 'g': g,
                                    'image': torch.where(Mf, inp['image'], state['image']), 'reg_in': (img_pw, dep_pw, feats_pw)}
        if self.reg_mode == 'crop':   # tile + margin crops of the patchwork (the TensorRT crop regressor's condition), cores written back
            mr = self.reg_crop
            cr = lambda x, s: crop_tiles(Ppx(x, s), idx, k * s, mr * s); cm = lambda x: x[..., 8 * mr:8 * (mr + k), 8 * mr:8 * (mr + k)]
            set_gn_mode(t.gs_parm_regresser, self.crop_norm, sample_ids=idx[:, 0])
            rot_p, scale_p, opac_p, res_p = t.gs_parm_regresser(cr(img_pw, 8), cr(dep_pw, 8), (cr(feats_pw[0], 4), cr(feats_pw[1], 2), cr(feats_pw[2], 1)))
            set_gn_mode(t.gs_parm_regresser, 'live', sample_ids=None)
            gp = state['g']
            g = {kk: Ppx(gp[kk]).clone() for kk in ('rot', 'scale', 'opacity', 'depth')}
            put_tiles(g['rot'], cm(rot_p), idx, Mcf); put_tiles(g['scale'], cm(scale_p), idx, Mcf); put_tiles(g['opacity'], cm(opac_p), idx, Mcf)
            put_tiles(g['depth'], crop_tiles(Ppx(depth), idx, 8 * k, 0) + cm(res_p), idx, Mcf)
            if self.tile_ctx: TileCtx.end()
            g = {kk: Upx(v) for kk, v in g.items()}
            g['img'] = torch.where(Mf, inp['image'], gp['img']); g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
            g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
            T['regressor'] = tick() - t0
            return {'g': g, 'flow': flow_u, 'mask': M.float(), 'changed': changed, 'stss_logits': stss_logits, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state_base(g)
        if self.tile_ctx: TileCtx.end()   # the other regressor modes run dense
        if self.reg_mode == 'dense_all':   # variant A: the regressor runs on the whole frame every frame (live statistics, as the per-frame recipe) and its output is used everywhere
            set_gn_mode(t.gs_parm_regresser, 'live')
            rot, scale, opacity, res = t.gs_parm_regresser(inp['image'], dep_pw, (feat[0], feat[1], U8(feat_cs)))
            g = {'rot': rot, 'scale': scale, 'opacity': opacity, 'depth': dep_pw + res, 'img': inp['image'], 'valid': (inp['mask'][:, :1] > 0.5).flatten(1)}
            g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
            T['regressor'] = tick() - t0
            st = new_state_base(g); st['reg_in'] = (inp['image'], dep_pw, (feat[0], feat[1], U8(feat_cs)))
            return {'g': g, 'flow': flow_u, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, st
        if self.reg_mode == 'dense_pw':   # dense module on the patchwork inputs, statistics of the start frame per view (= what DeltaCNN computes)
            set_gn_mode(t.gs_parm_regresser, self.crop_norm)
            rot, scale, opacity, res = t.gs_parm_regresser(img_pw, dep_pw, feats_pw)
            set_gn_mode(t.gs_parm_regresser, 'live')
        else:   # 'delta'
            raw = torch.cat([self.dc[v](cl(img_pw[v:v + 1]), cl(dep_pw[v:v + 1]), [cl(x[v:v + 1]) for x in feats_pw]) for v in range(2)], 0)
            rot, scale, opacity, res = head_activations(t.gs_parm_regresser, raw)
        gp = state['g']
        g = {'rot': torch.where(Mf, rot, gp['rot']), 'scale': torch.where(Mf, scale, gp['scale']), 'opacity': torch.where(Mf, opacity, gp['opacity']),
             'depth': torch.where(Mf, depth + res, gp['depth']), 'img': torch.where(Mf, inp['image'], gp['img']), 'valid': (inp['mask'][:, :1] > 0.5).flatten(1)}
        g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
        T['regressor'] = tick() - t0
        return {'g': g, 'flow': flow_u, 'mask': M.float(), 'changed': changed, 'stss_logits': stss_logits, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state_base(g)
