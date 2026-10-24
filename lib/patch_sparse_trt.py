import os
import time
import torch
import torch.nn.functional as F
from core.utils.utils import downflow8
from lib.utils import depth2pc
from lib.temporal_gps import TemporalGPS
from lib.temporal_sparse import crop_tiles
from lib.temporal_patch import put_tiles, patch_tiles
from lib.patch_step3 import PatchStep3
from lib.patch_sparse import PatchSparse, att_sparse
from lib.trt_sparse import TrtNet, affine_layout, affine_vectors
from lib.tile_mode import TileCtx, convex_upsample_ring, ring_groups, gather_rings, write_tiles_padded, BandStore

# The tile-cell execution with TensorRT engines (lib/trt_sparse.py, build_sparse_trt.py): the encoder engine is dense
# (step 1; STSS needs the whole 1/8 map), STSS and the query-sparse epipolar attention run in PyTorch, then three crop
# engines (attention residual block + cnet, GRU x 8 + upsampling, regressor) run on the N tile crops of the frame
# (chunks of at most max_n), and only the tile cores are written back. Normalisation statistics of the start frame go in
# as an input per crop (its view). Start frame: dense, from `key_model` (the frozen original network) as in PatchSparse.


class PatchSparseTRT(PatchSparse):
    def __init__(self, model: TemporalGPS, engine_dir, patch=128, halo=4, reg_margin=4, key_model=None, enc_engine='experiments/gps_plus_dynerf_jit2_1007/trt/sparse/enc_fp16.engine', precision='fp16', max_n=96, depth_sparse=True, reg_mode='crop', tile_ctx=False, graph=False):
        super().__init__(model, patch, halo, key_model=key_model, dense_reg=True, reg_crop=reg_margin, depth_sparse=depth_sparse, reg_mode=reg_mode, tile_ctx=tile_ctx)   # no DeltaCNN instances (the regressor is an engine)
        t = model.teacher
        self.mr, self.max_n = reg_margin, max_n
        k, tt, S = self.k, self.k + 2 * halo, 8 * (self.k + 2 * reg_margin)
        e = lambda n: os.path.join(engine_dir, '%s_%s.engine' % (n, precision))
        self.enc = TrtNet(enc_engine, ['image'], ['f1', 'f2', 'f3'])
        if tile_ctx:   # variant D: ring-input engines (lib/tile_mode.py); one ring per convolution / upsampling from the per-layer stores
            import json
            # the engines keep the engine's own I/O names: rings of layers whose output is never used (e.g. the mask head of the GRU's
            # first 7 iterations) were pruned by the export and are simply not gathered
            # grouped rings (rings_g.json, lib/tile_mode.ring_groups): one ring input / tile-input output per (store width, ring) group
            gpath, bpath = os.path.join(engine_dir, 'rings_g.json'), os.path.join(engine_dir, 'rings_b.json')
            self.ring_b = json.load(open(bpath))['groups'] if os.path.exists(bpath) else None   # band stores (ring strips in, border strips out)
            self.ring_g = json.load(open(gpath))['groups'] if os.path.exists(gpath) else None   # full ring windows per group
            sfx = 'ringb' if self.ring_b is not None else 'ringg' if self.ring_g is not None else 'ring'
            self.front = TrtNet(e('front_%s%d' % (sfx, tt)), graph=graph); self.front.stage_in, self.front.stage_out = ['att2', 'aff'], ['fcs', 'fm', 'net', 'ctx']
            self.gru = TrtNet(e('gru_%s%d' % (sfx, tt)), graph=graph); self.gru.stage_in, self.gru.stage_out = ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'], ['flow', 'mask']
            self.regr_ring = TrtNet(e('regr_%s%d' % (sfx, S)), graph=graph); self.regr_ring.stage_in, self.regr_ring.stage_out = ['img', 'dep', 'f1', 'f2', 'f3', 'aff'], ['rot', 'scale', 'opacity', 'res']
        else:
            self.front = TrtNet(e('front_crop%d' % tt), ['att2', 'aff'], ['fcs', 'fm', 'net', 'ctx'], graph=graph)
            self.gru = TrtNet(e('gru_crop%d' % tt), ['fm', 'rows', 'net', 'ctx', 'c0x', 'fi'], ['up'], graph=graph)
        self.regr = TrtNet(e('regr_crop%d' % S), ['img', 'dep', 'f1', 'f2', 'f3', 'aff'], ['rot', 'scale', 'opacity', 'res'], graph=graph) if (reg_mode == 'crop' and not tile_ctx) else None
        self.regr_dense = TrtNet(e('regr'), ['image', 'depth', 'f1', 'f2', 'fcs'], ['rot', 'scale', 'opacity', 'res']) if reg_mode == 'dense_all' else None   # variant A: dense regressor, live statistics
        if not depth_sparse:   # variant B: dense depth engines (encoder + attention + residual block + cnet; GRU x 8 + upsampling)
            self.front_dense = TrtNet(e('front'), ['image'], ['f1', 'f2', 'f3', 'att1', 'att2', 'fcs', 'fm', 'net', 'ctx'])
            self.gru_dense = TrtNet(e('gru'), ['fm', 'net', 'ctx', 'fi'], ['up'])
        self.att = [TrtNet(e('att_rows_l%d' % i), ['x', 'src'], ['out']) for i in range(2)] if os.path.exists(e('att_rows_l0')) else None   # row-batched attention engines (tile rows only)
        self.gstores = self.bstores = None   # grouped ring stores / band stores (variant D), built on the start frame
        self.lay_front, self.P_front = affine_layout([t.loftr_coarse.resid, self.rs.cnet])
        self.lay_regr, self.P_regr = affine_layout([t.gs_parm_regresser])
        self.aff_front = self.aff_regr = None

    @torch.no_grad()
    def keyframe(self, data):
        st = super().keyframe(data)   # records the statistics of `model` on the start frame, stored maps from key_model
        self.aff_front, self.aff_regr = affine_vectors(self.lay_front, self.P_front), affine_vectors(self.lay_regr, self.P_regr)
        if self.tile_ctx and self.ring_b is not None:
            self._band_stores()
        elif self.tile_ctx and self.ring_g is not None:
            self._group_stores()
        return st

    def _band_stores(self):
        """band stores (lib/tile_mode.BandStore) per (store width, ring) group of a stage from the recorded full-frame maps, which are
        then dropped (the band stores hold everything the tile rings ever read)"""
        G = ring_groups(self.stage_keys, self.meta, self.stores)
        self.bstores = {}
        for stage, groups in G.items():
            spec = self.ring_b[stage]
            assert [(g['sw'], g['r'], g['C']) for g in groups] == [(d['sw'], d['r'], d['C']) for d in spec], 'ring groups differ from the engine build: %s' % stage
            keys, self.bstores[stage] = self.stage_keys[stage], []
            for g in groups:
                B, _, H, W = self.stores[keys[g['layers'][0][0]]].shape
                bs = BandStore(B, g['C'], H, W, self.k * W // 128, g['r'])
                for i, off, C in g['layers']:
                    assert self.stores[keys[i]].shape[-2:] == (H, W)
                    bs.fill(self.stores[keys[i]], off); del self.stores[keys[i]]
                self.bstores[stage].append(bs)
        torch.cuda.empty_cache()

    def _group_stores(self):
        """grouped rings: one padded store per (store width, ring) group of a stage, the layers' channels in call order (the layout the
        engine was built with); the per-layer stores become views into the interiors"""
        G = ring_groups(self.stage_keys, self.meta, self.stores)
        self.gstores = {}
        for stage, groups in G.items():
            spec = self.ring_g[stage]
            assert [(g['sw'], g['r'], g['C']) for g in groups] == [(d['sw'], d['r'], d['C']) for d in spec], 'ring groups differ from the engine build: %s' % stage
            keys, self.gstores[stage] = self.stage_keys[stage], []
            for g in groups:
                B, _, H, W = self.stores[keys[g['layers'][0][0]]].shape; r = g['r']
                gs = torch.zeros(B, g['C'], H + 2 * r, W + 2 * r, device='cuda')
                for i, off, C in g['layers']:
                    assert self.stores[keys[i]].shape[-2:] == (H, W)
                    gs[:, off:off + C, r:r + H, r:r + W] = self.stores[keys[i]]
                    self.stores[keys[i]] = gs[:, off:off + C, r:r + H, r:r + W]
                self.gstores[stage].append((gs, r, W))

    def _att(self, i, x, src, cells):
        """layer i on the rows that contain tile cells (engine), returns the new values of the tile cells as att_sparse does"""
        if self.att is None:
            return att_sparse(self.m.teacher.loftr_coarse.layers[i], x, src, cells)
        C, H, W = x.shape[1:]
        ys, xs = torch.where(cells)
        rows = torch.unique(ys)
        out_rows = self.att[i](x[0].permute(1, 2, 0)[rows], src[0].permute(1, 2, 0)[rows])[0]     # (R, W, C)
        order = torch.argsort(ys * W + xs); ys_o, xs_o = ys[order], xs[order]
        return out_rows[torch.searchsorted(rows, ys_o), xs_o], ys_o, xs_o

    def _ring(self, stage, net, idx, *stage_inputs):
        """variant D: gather one ring per layer from the stores (call order of the record pass), run the ring engine, write the tile inputs back"""
        if self.bstores is not None:
            return self._ring_b(stage, net, idx, *stage_inputs)
        if self.gstores is not None:
            return self._ring_g(stage, net, idx, *stage_inputs)
        keys = self.stage_keys[stage]; n = stage_inputs[0].shape[0]
        kw = dict(zip(net.stage_in, stage_inputs))
        for i, key in enumerate(keys):
            name = 'ring_%d' % i
            if name in net.inputs:
                r, sw = self.meta[key]; s_t = self.k * sw // 128
                kw[name] = crop_tiles(self.stores[key], idx, s_t, r)
        outs = net.call_named(**kw)
        for i, key in enumerate(keys):
            x = outs.get('xin_%d' % i)
            if x is not None:
                s_t = x.shape[-1]
                put_tiles(self.stores[key], x, idx, torch.ones(n, 1, s_t, s_t, dtype=torch.bool, device=x.device))
        return [outs[nm].clone() for nm in net.stage_out]

    def _ring_g(self, stage, net, idx, *stage_inputs):
        """grouped rings: one strided gather per group for all tiles of the frame (context = the stores before this frame's updates, as in
        the PyTorch execution), engine in chunks of max_n, one write-back per group"""
        n = stage_inputs[0].shape[0]
        rings = [gather_rings(gs, idx, self.k * W // 128, r) for gs, r, W in self.gstores[stage]]
        outs, tiles = None, [[] for _ in rings]
        for c0 in range(0, n, self.max_n):
            sl = slice(c0, c0 + self.max_n)
            kw = dict(zip(net.stage_in, [x[sl] for x in stage_inputs])); kw.update({'ring_g%d' % j: rg[sl] for j, rg in enumerate(rings)})
            o = net.call_named(**kw)
            part = [o[nm].clone() for nm in net.stage_out]
            outs = part if outs is None else [torch.cat([a, b]) for a, b in zip(outs, part)]
            for j in range(len(rings)): tiles[j].append(o['xin_g%d' % j].clone())
        for j, (gs, r, W) in enumerate(self.gstores[stage]):
            write_tiles_padded(gs, tiles[j][0] if len(tiles[j]) == 1 else torch.cat(tiles[j]), idx, r)
        return outs

    def _ring_b(self, stage, net, idx, *stage_inputs):
        """band stores: per group one gather of the ring strips for all tiles of the frame (context = the stores before this frame's
        updates), engine in chunks of max_n, one write of the fresh tiles' border strips per group"""
        n = stage_inputs[0].shape[0]
        strips = [bs.gather(idx) for bs in self.bstores[stage]]
        outs, parts = None, [[] for _ in range(2 * len(strips))]
        for c0 in range(0, n, self.max_n):
            sl = slice(c0, c0 + self.max_n)
            kw = dict(zip(net.stage_in, [x[sl] for x in stage_inputs]))
            for j, (tb, lr) in enumerate(strips): kw['tb_g%d' % j], kw['lr_g%d' % j] = tb[sl], lr[sl]
            o = net.call_named(**kw)
            part = [o[nm].clone() for nm in net.stage_out]
            outs = part if outs is None else [torch.cat([a, b]) for a, b in zip(outs, part)]
            for j in range(len(strips)): parts[2 * j].append(o['xtb_g%d' % j].clone()); parts[2 * j + 1].append(o['xlr_g%d' % j].clone())
        cat = lambda l: l[0] if len(l) == 1 else torch.cat(l)
        for j, bs in enumerate(self.bstores[stage]):
            bs.write(idx, cat(parts[2 * j]), cat(parts[2 * j + 1]))
        return outs

    def _run(self, net, *args):
        """engine on N crops in chunks of max_n; returns the outputs concatenated (copies)"""
        n = args[0].shape[0]
        if n <= self.max_n:
            return [o.clone() for o in net(*args)]
        outs = None
        for i in range(0, n, self.max_n):
            part = [o.clone() for o in net(*[x[i:i + self.max_n] for x in args])]
            outs = part if outs is None else [torch.cat([a, b]) for a, b in zip(outs, part)]
        return outs

    def _step_dense_depth_trt(self, data, state, inp, changed, M, Mf, Mp, idx, T, tick, t0):
        """variant B: dense depth engines every frame (front: encoder+attention+residual+cnet; gru), regressor on the tile crops (engine), tiles written back"""
        m = self.m; k, mr = self.k, self.mr
        f1, f2, f3, att1, att2, fcs, fm, net, ctx = self._front_out   # computed once at the top of step
        up, = self.gru_dense(fm, net, ctx, downflow8(inp['flow_init']))
        flow = up.clone()
        T['attention'] = tick() - t0; t0 = tick(); T['front'] = 0.0; T['gru'] = 0.0
        depth = m.to_depth(flow, inp)
        H, W = inp['image'].shape[-2:]; h8 = f3.shape[-2]; pad = self._pads(h8); Ppx = lambda x, s=8: self._ppx(x, pad, s)
        tt = k
        new_state_base = lambda g: {'f3': torch.where(M, f3, state['f3']), 'att1': att1, 'att2': att2, 'feat_cs': fcs, 'fmap': fm, 'ctx': list(ctx.split(ctx.shape[1] // 3, 1)), 'flow': flow, 'g': g,
                                    'image': inp['image'], 'reg_in': (inp['image'], depth, (f1, f2, fcs))}
        if idx.shape[0] == 0:
            T['regressor'] = 0.0
            return {'g': state['g'], 'flow': flow, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': 0}, new_state_base(state['g'])
        Mc = crop_tiles(Mp, idx, k, 0); Mcf = F.interpolate(Mc.float(), scale_factor=8, mode='nearest') > 0.5
        views = idx[:, 0]
        cr = lambda x, s: crop_tiles(Ppx(x, s), idx, k * s, mr * s); cm = lambda x: x[..., 8 * mr:8 * (mr + k), 8 * mr:8 * (mr + k)]
        rot_p, scale_p, opac_p, res_p = self._run(self.regr, cr(inp['image'], 8), cr(depth, 8), cr(f1, 4), cr(f2, 2), cr(fcs, 1), self.aff_regr[views])
        gp = state['g']
        g = {kk: Ppx(gp[kk]).clone() for kk in ('rot', 'scale', 'opacity', 'depth')}
        put_tiles(g['rot'], cm(rot_p), idx, Mcf); put_tiles(g['scale'], cm(scale_p), idx, Mcf); put_tiles(g['opacity'], cm(opac_p), idx, Mcf)
        put_tiles(g['depth'], crop_tiles(Ppx(depth), idx, 8 * k, 0) + cm(res_p), idx, Mcf)
        g = {kk: self._u(v, H, W) for kk, v in g.items()}
        g['img'] = torch.where(Mf, inp['image'], gp['img']); g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
        g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
        T['regressor'] = tick() - t0
        return {'g': g, 'flow': flow, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state_base(g)

    @torch.no_grad()
    def step(self, data, state):
        m, t = self.m, self.m.teacher
        k, halo, mr = self.k, self.halo, self.mr
        T = {}; tick = lambda: (torch.cuda.synchronize(), time.perf_counter())[1]
        t0 = tick()
        inp = m.inputs(data)
        if not self.depth_sparse:   # variant B: the dense front engine (encoder + attention + residual block + cnet) gives f3 for STSS as well
            self._front_out = [x.clone() for x in self.front_dense(inp['image'])]
            f1, f2, f3 = self._front_out[:3]
        else:
            f1, f2, f3 = [x.clone() for x in self.enc(inp['image'])]          # dense encoder engine (STSS needs the whole map)
        h8, w8 = f3.shape[-2:]; H, W = inp['image'].shape[-2:]
        pad = self._pads(h8); P8 = lambda x: self._p8(x, pad); Ppx = lambda x, s=8: self._ppx(x, pad, s); U8 = lambda x: self._u(x, h8, w8); Upx = lambda x: self._u(x, H, W)
        T['encoder'] = tick() - t0; t0 = tick()
        changed = (torch.sigmoid(m.stss(f3, state['f3'])) > 0.5).float()
        M = PatchStep3.footprint_mask(changed, k, 'tile', 0)
        Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
        Mp = P8(M)
        idx = patch_tiles(Mp, k)
        T['stss'] = tick() - t0; t0 = tick()
        if idx.shape[0] == 0:
            for key in ('attention', 'front', 'gru', 'regressor'): T[key] = 0.0
            return {'g': state['g'], 'flow': state['flow'], 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': 0}, dict(state)
        if not self.depth_sparse:
            return self._step_dense_depth_trt(data, state, inp, changed, M, Mf, Mp, idx, T, tick, t0)
        # ---- epipolar attention at the tile cells (rows with tiles; engine or PyTorch)
        f3p = P8(f3)
        a1 = [P8(state['att1'][v:v + 1]).clone() for v in (0, 1)]
        for v in (0, 1):
            if M[v, 0].any():
                src = f3p[1 - v:2 - v] if v == 0 else a1[0]
                out, ys, xs = self._att(0, f3p[v:v + 1], src, Mp[v, 0]); a1[v][0, :, ys, xs] = out.t()
        a2 = [P8(state['att2'][v:v + 1]).clone() for v in (0, 1)]
        for v in (0, 1):
            if M[v, 0].any():
                src = a1[1] if v == 0 else a2[0]
                out, ys, xs = self._att(1, a1[v], src, Mp[v, 0]); a2[v][0, :, ys, xs] = out.t()
        att1, att2 = torch.cat(a1), torch.cat(a2)
        T['attention'] = tick() - t0; t0 = tick()
        # ---- attention residual block + context net on the crops (engine)
        crop = lambda x, s=1: crop_tiles(x, idx, k * s, halo * s)
        core = lambda x, s=1: x[..., halo * s:(halo + k) * s, halo * s:(halo + k) * s]
        Mc = crop(Mp)[..., halo:halo + k, halo:halo + k]
        Mcf = F.interpolate(Mc.float(), scale_factor=8, mode='nearest') > 0.5
        views = idx[:, 0]
        fcs_p, fm_p, net_p, ctx_p = self._ring('front', self.front, idx, crop(att2), self.aff_front[views]) if self.tile_ctx else self._run(self.front, crop(att2), self.aff_front[views])
        feat_cs = P8(state['feat_cs']).clone(); put_tiles(feat_cs, core(fcs_p), idx, Mc)
        fmap = P8(state['fmap']).clone(); put_tiles(fmap, core(fm_p), idx, Mc)
        ctx_l = [P8(c).clone() for c in state['ctx']]
        for c, cp in zip(ctx_l, ctx_p.split(ctx_p.shape[1] // 3, 1)): put_tiles(c, core(cp), idx, Mc)
        T['front'] = tick() - t0; t0 = tick()
        # ---- cost volume against the other view's full rows, GRU x 8, upsampling (engine)
        tt = k + 2 * halo
        fo = F.pad(fmap, (0, 0, halo, halo)); ar = torch.arange(tt, device=f3.device)
        Y = (idx[:, 1] * k)[:, None] + ar[None, :]
        rows = fo[(1 - views)[:, None], :, Y].permute(0, 2, 1, 3).contiguous()
        if self.coords0.shape[-1] != w8 + pad:
            from core.utils.utils import coords_grid
            self.coords0 = coords_grid(2, h8 + pad, w8 + pad).to(f3.device)
        if self.tile_ctx:
            flow_lr, up_mask = self._ring('gru', self.gru, idx, fm_p, rows, net_p, ctx_p, crop(self.coords0[:, :1]), crop(downflow8(Ppx(inp['flow_init']))))
            TileCtx.idx, TileCtx.write = idx, True
            up_p = convex_upsample_ring(torch.cat([flow_lr, torch.zeros_like(flow_lr)], 1), up_mask, self.stores['flow8'])[:, :1]   # the store holds both flow channels
        else:
            up_p, = self._run(self.gru, fm_p, rows, net_p, ctx_p, crop(self.coords0[:, :1]), crop(downflow8(Ppx(inp['flow_init']))))
        flow = Ppx(state['flow']).clone(); put_tiles(flow, core(up_p, 8), idx, Mcf)
        flow_u = Upx(flow)
        T['gru'] = tick() - t0; t0 = tick()
        # ---- regressor on tile crops (+ margin) of the patchwork inputs (engine), cores written back
        depth = m.to_depth(flow_u, inp)
        img0, dep0, f0 = state['reg_in']
        img_pw = torch.where(Mf, inp['image'], img0); dep_pw = torch.where(Mf, depth, dep0)
        up_s = lambda s: F.interpolate(M.float(), scale_factor=8 // s, mode='nearest') > 0.5
        feats_pw = [torch.where(up_s(inp['image'].shape[-1] // x.shape[-1]), x, r) for x, r in zip((f1, f2, U8(feat_cs)), f0)]
        if self.reg_mode == 'dense_all':   # variant A: the dense regressor engine on the whole frame, output used everywhere
            rot, scale, opacity, res = [x.clone() for x in self.regr_dense(inp['image'], dep_pw, f1, f2, U8(feat_cs))]
            g = {'rot': rot, 'scale': scale, 'opacity': opacity, 'depth': dep_pw + res, 'img': inp['image'], 'valid': (inp['mask'][:, :1] > 0.5).flatten(1)}
            g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
            T['regressor'] = tick() - t0
            new_state = {'f3': torch.where(M, f3, state['f3']), 'att1': U8(att1), 'att2': U8(att2), 'feat_cs': U8(feat_cs), 'fmap': U8(fmap), 'ctx': [U8(c) for c in ctx_l], 'flow': flow_u, 'g': g,
                         'image': inp['image'], 'reg_in': (inp['image'], dep_pw, (f1, f2, U8(feat_cs)))}
            return {'g': g, 'flow': flow_u, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state
        cr = lambda x, s: crop_tiles(Ppx(x, s), idx, k * s, mr * s)
        if self.tile_ctx:
            rot_p, scale_p, opac_p, res_p = self._ring('regr', self.regr_ring, idx, cr(img_pw, 8), cr(dep_pw, 8), cr(feats_pw[0], 4), cr(feats_pw[1], 2), cr(feats_pw[2], 1), self.aff_regr[views])
        else:
            rot_p, scale_p, opac_p, res_p = self._run(self.regr, cr(img_pw, 8), cr(dep_pw, 8), cr(feats_pw[0], 4), cr(feats_pw[1], 2), cr(feats_pw[2], 1), self.aff_regr[views])
        cm = lambda x: x[..., 8 * mr:8 * (mr + k), 8 * mr:8 * (mr + k)]
        gp = state['g']
        g = {kk: Ppx(gp[kk]).clone() for kk in ('rot', 'scale', 'opacity', 'depth')}
        put_tiles(g['rot'], cm(rot_p), idx, Mcf); put_tiles(g['scale'], cm(scale_p), idx, Mcf); put_tiles(g['opacity'], cm(opac_p), idx, Mcf)
        put_tiles(g['depth'], crop_tiles(Ppx(depth), idx, 8 * k, 0) + cm(res_p), idx, Mcf)   # tile core of depth + residual
        g = {kk: Upx(v) for kk, v in g.items()}
        g['img'] = torch.where(Mf, inp['image'], gp['img']); g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
        g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
        T['regressor'] = tick() - t0
        new_state = {'f3': torch.where(M, f3, state['f3']), 'att1': U8(att1), 'att2': U8(att2), 'feat_cs': U8(feat_cs), 'fmap': U8(fmap), 'ctx': [U8(c) for c in ctx_l], 'flow': flow_u, 'g': g,
                     'image': torch.where(Mf, inp['image'], state['image']), 'reg_in': (img_pw, dep_pw, feats_pw)}
        return {'g': g, 'flow': flow_u, 'mask': M.float(), 'changed': changed, 'times': T, 'crop_cells': idx.shape[0] * tt * tt}, new_state
