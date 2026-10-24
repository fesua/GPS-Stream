import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import coords_grid
from lib.temporal_gps import lookup

# Sparse execution of TemporalGPS.step: only the changed region is computed.
#   point-wise parts (proposal MLP, local epipolar attention, local cost volume) run on the changed positions only;
#   conv parts (attention residual block, cnet, GRU, upsampling, Gaussian regressor) run once on the changed region
#   plus the context their receptive field needs: the connected parts of (changed region + margin) as merged
#   bounding boxes, results are written to the changed cells only.
# Normalisation layers use full-map statistics recorded once per keyframe (FrozenNorm), so the sparse result
# equals the dense step with the same statistics.

CELL = 8      # feature-map cell = 8x8 pixels
TILE = 16     # tile size in cells (128 px)


class FrozenNorm(nn.Module):
    """GroupNorm / InstanceNorm with full-map statistics: 'live' computes them, 'record' computes and keeps
    them, 'frozen' uses the kept ones (per view; view_ids maps crop batches to views)"""
    view_ids = None

    def __init__(self, norm):
        super().__init__()
        self.groups = norm.num_groups if isinstance(norm, nn.GroupNorm) else norm.num_features
        self.eps = norm.eps
        self.weight, self.bias = norm.weight, norm.bias
        self.mode, self.mean, self.var = 'live', None, None

    def forward(self, x):
        N, C, H, W = x.shape
        xg = x.reshape(N, self.groups, -1)
        if self.mode == 'frozen':
            ids = FrozenNorm.view_ids if FrozenNorm.view_ids is not None else torch.arange(N, device=x.device)
            mean, var = self.mean[ids], self.var[ids]
        else:
            mean, var = xg.mean(-1, keepdim=True), xg.var(-1, unbiased=False, keepdim=True)
            if self.mode == 'record':
                self.mean, self.var = mean.detach(), var.detach()
        y = ((xg - mean) / torch.sqrt(var + self.eps)).reshape(N, C, H, W)
        if self.weight is not None:
            y = y * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return y


def freeze_norms(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.GroupNorm, nn.InstanceNorm2d)):
            setattr(module, name, FrozenNorm(child))
        else:
            freeze_norms(child)
    return module


def set_norm_mode(module, mode):
    for m in module.modules():
        if isinstance(m, FrozenNorm):
            m.mode = mode


# ---------------- tiles ----------------
def dilate(mask, r):
    return F.max_pool2d(mask.float(), 2 * r + 1, 1, r) > 0 if r > 0 else mask


def tiles_of(mask, tile):
    """(n, 3) = view, tile row, tile col of the tiles that contain a True cell"""
    t = F.max_pool2d(mask.float(), tile, tile) > 0
    return t[:, 0].nonzero()


def crop_tiles(x, idx, tile, halo):
    """x (B, C, H, W) -> (n, C, tile + 2 halo, tile + 2 halo), zeros outside the image"""
    xp = F.pad(x, (halo, halo, halo, halo))
    ar = torch.arange(tile + 2 * halo, device=x.device)
    Y = (idx[:, 1] * tile)[:, None, None] + ar[None, :, None]
    X = (idx[:, 2] * tile)[:, None, None] + ar[None, None, :]
    return xp[idx[:, 0][:, None, None], :, Y, X].permute(0, 3, 1, 2).contiguous()


def write_tiles(dense, vals, idx, tile, mask):
    """dense[mask] <- centre tiles vals (n, C, tile, tile) where mask (B, 1, H, W)"""
    ar = torch.arange(tile, device=dense.device)
    Y = ((idx[:, 1] * tile)[:, None, None] + ar[None, :, None]).expand(-1, tile, tile)
    X = ((idx[:, 2] * tile)[:, None, None] + ar[None, None, :]).expand(-1, tile, tile)
    B = idx[:, 0][:, None, None].expand(-1, tile, tile)
    m = mask[B, 0, Y, X]
    dense[B[m], :, Y[m], X[m]] = vals.permute(0, 2, 3, 1)[m]


def centre(x, halo):
    return x[..., halo:x.shape[-2] - halo, halo:x.shape[-1] - halo]


def region_boxes(mask, margin, quant=4):
    """bounding boxes (view, y0, y1, x0, x1) of the connected parts of dilate(mask, margin); overlapping boxes are
    merged. Sides are rounded up to multiples of quant cells so that box shapes repeat across frames (cuDNN builds
    a convolution plan per input shape)"""
    from scipy import ndimage
    D = dilate(mask, margin)[:, 0].cpu().numpy()
    H, W = D.shape[1:]

    def grow(lo, hi, size):
        n = min(-(-(hi - lo) // quant) * quant, size)
        lo = min(max(lo - (n - (hi - lo)) // 2, 0), size - n)
        return lo, lo + n

    out = []
    for b in range(D.shape[0]):
        boxes = [[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop] for sl in ndimage.find_objects(ndimage.label(D[b])[0])]
        changed = True
        while changed:
            changed = False
            boxes = [[*grow(y0, y1, H), *grow(x0, x1, W)] for y0, y1, x0, x1 in boxes]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, c = boxes[i], boxes[j]
                    if a[0] < c[1] and c[0] < a[1] and a[2] < c[3] and c[2] < a[3]:
                        boxes[i] = [min(a[0], c[0]), max(a[1], c[1]), min(a[2], c[2]), max(a[3], c[3])]
                        boxes.pop(j)
                        changed = True
                        break
                if changed:
                    break
        out += [(b, *bx) for bx in boxes]
    return out


# ---------------- point-wise parts ----------------
def sample_row_pts(feat, b, y, xs):
    """feat (B, C, H, W), points (b, y) and x-positions xs (N, K) on their row -> (N, K, C), zero outside"""
    W = feat.shape[-1]
    x0 = torch.floor(xs)
    out = 0
    for xi, wi in [(x0, 1 - (xs - x0)), (x0 + 1, xs - x0)]:
        valid = (xi >= 0) & (xi <= W - 1)
        xc = xi.clamp(0, W - 1).long()
        out = out + feat[b[:, None].expand_as(xc), :, y[:, None].expand_as(xc), xc] * (wi * valid)[..., None]
    return out


def mlp_pts(seq, x):
    for m in seq:
        x = F.linear(x, m.weight.flatten(1), m.bias) if isinstance(m, nn.Conv2d) else m(x)
    return x


def local_attention_pts(layer, xq, keys):
    """Crossatt_EncoderLayer for N queries (N, C) with their own keys (N, K, C)"""
    att = layer.attention
    C = xq.shape[1]
    nh, hd = att.num_heads, C // att.num_heads
    q = layer.q_proj(layer.norm1(xq))
    src = layer.norm1(keys)
    k, v = layer.k_proj(src), layer.v_proj(src)
    wq, wk, wv = att.in_proj_weight.chunk(3)
    bq, bk, bv = att.in_proj_bias.chunk(3)
    Q = F.linear(q, wq, bq).view(-1, nh, hd)
    K = F.linear(k, wk, bk).view(k.shape[0], k.shape[1], nh, hd)
    V = F.linear(v, wv, bv).view(k.shape[0], k.shape[1], nh, hd)
    a = torch.softmax(torch.einsum('nhd,nkhd->nhk', Q, K) / math.sqrt(hd), -1)
    m = layer.merge(att.out_proj(torch.einsum('nhk,nkhd->nhd', a, V).reshape(-1, C)))
    return xq + layer.norm2(layer.mlp(torch.cat([xq, m], 1)))


class SparseTemporalGPS:
    def __init__(self, model, margin=8, patch=None, bucket=32):
        """model: a trained TemporalGPS; its normalisation layers are switched to FrozenNorm.
        margin: context (cells) around the changed region; the GRU delta reads +-8 cells, the regressor ~64 px.
        patch: (k, halo) -> conv parts run on k x k-cell tiles with a halo (lib/temporal_patch.py) instead of boxes;
        the patch batch is padded to a multiple of bucket so that shapes repeat"""
        self.m = model
        for mod in [model.att_resid, model.cnet, model.regressor]:
            freeze_norms(mod)
        self.r = model.r
        self.margin = margin
        self.patch, self.bucket = patch, bucket
        self.timing = None

    # ---- module calls on the regions (overridden by the TensorRT runner) ----
    def encode(self, image):
        return self.m.teacher.img_encoder(image)

    def stss_logits(self, f3, f3_prev):
        return self.m.stss(f3, f3_prev)

    def resid(self, x, b):
        FrozenNorm.view_ids = torch.tensor([b], device=x.device)
        return self.m.att_resid(x)

    def cnet(self, x, b):
        """feature head (+-3 cells) and context through the zqr conv (+-4 cells) -> fmap, [cz, cr, cq]"""
        m = self.m
        FrozenNorm.view_ids = torch.tensor([b], device=x.device)
        zq = m.zqr[0](torch.relu(m.cnet.outputs08[1](x)))
        return m.cnet.conv2(x), list(zq.split(zq.shape[1] // 3, 1))

    def gru(self, net, ctx, cv, centers, coords1, coords0, M):
        """all GRU iterations on one region; updates only the cells in M; returns net, coords1, upsampled flow"""
        m = self.m
        Mf = M.float()
        for _ in range(m.iters):
            corr = lookup(cv, centers, coords1[:, :1], self.r)
            nl, up_mask, delta = m.update_block([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
            net = torch.where(M, nl[0], net)
            coords1 = coords1 + torch.cat([delta[:, :1], torch.zeros_like(delta[:, 1:])], 1) * Mf
        return net, coords1, m.upsample(coords1 - coords0, up_mask)[:, :1]

    def regress(self, img, depth, f1, f2, fcs, b):
        FrozenNorm.view_ids = torch.tensor([b], device=img.device)
        return self.m.regressor(img, depth, (f1, f2, fcs))

    def _t(self, name):
        if self.timing is not None:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.timing[name] = self.timing.get(name, 0) + (now - self._last) * 1000
            self._last = now

    # ---------------- keyframe ----------------
    @torch.no_grad()
    def keyframe(self, data):
        """per-frame model + one dense pass of the temporal modules that records the normalisation statistics
        and fills the cnet caches"""
        m = self.m
        state = m.keyframe(data)
        set_norm_mode(m, 'record')
        mask = torch.zeros_like(state['centers'])
        out, _ = m.step(data, state, mask=mask)
        set_norm_mode(m, 'frozen')
        if getattr(m, 'full_attention', False):
            set_norm_mode(m.cnet, 'live')  # cnet runs on the whole map every frame: live full-map statistics
        state['fmap'], state['ctx'] = out['fmap'], [c.clone() for c in out['ctx']]
        state['coords0'] = coords_grid(2, *state['centers'].shape[-2:]).to(state['centers'].device)
        return state

    @torch.no_grad()
    def warmup_patches(self, data, state):
        """patch mode: runs the conv parts once for every padded batch size, up to all tiles of both views"""
        from lib.temporal_patch import patch_convs
        m, (k, halo) = self.m, self.patch
        inp = m.inputs(data)
        feat = self.encode(inp['image'])
        B, _, H, W = state['centers'].shape
        tiles = torch.stack(torch.meshgrid(torch.arange(B), torch.arange(H // k), torch.arange(W // k), indexing='ij'), -1).view(-1, 3).to(feat[2].device)
        M = torch.zeros_like(state['centers'], dtype=torch.bool)
        for n in range(self.bucket, tiles.shape[0] + 1, self.bucket):
            patch_convs(m, M, tiles[:1], k, halo, state['feat_cs'], state['centers'], feat, inp, state, pad_to=n)
        torch.cuda.synchronize()

    @torch.no_grad()
    def warmup(self, data, state, quant=4):
        """runs the region modules once on every box shape region_boxes can produce (sides quant..H cells in steps of
        quant), so that no convolution meets a new input shape while streaming (cuDNN builds a plan per shape)"""
        m = self.m
        inp = m.inputs(data)
        feat = self.encode(inp['image'])
        H, W = state['centers'].shape[-2:]
        cam = {k: inp[k][0:1] for k in ('intr', 'ref_intr', 'tf_x', 'extr')}
        for h in range(quant, H + 1, quant):
            for w in range(quant, W + 1, quant):
                sl = (slice(0, 1), slice(None), slice(0, h), slice(0, w))
                slf = (slice(0, 1), slice(None), slice(0, h * CELL), slice(0, w * CELL))
                s2 = (slice(0, 1), slice(None), slice(0, h * 2), slice(0, w * 2))
                s4 = (slice(0, 1), slice(None), slice(0, h * 4), slice(0, w * 4))
                if not getattr(m, 'full_attention', False):  # full-attention models run these on the whole map
                    self.resid(state['feat_cs'][sl], 0)
                    self.cnet(state['feat_cs'][sl], 0)
                self.gru(state['net'][sl], [c[sl] for c in state['ctx']], state['cv'][sl], state['centers'][sl],
                         state['coords1'][sl], state['coords0'][sl], torch.ones_like(state['centers'][sl], dtype=torch.bool))
                self.regress(inp['image'][slf], m.to_depth(state['flow'][slf], cam), feat[0][s4], feat[1][s2],
                             state['feat_cs'][sl], 0)
        FrozenNorm.view_ids = None
        torch.cuda.synchronize()

    def _step_full(self, inp, feat, state, st, M, changed):
        """full_attention model: the original epipolar attention and cnet on the whole map (current frame, both views),
        the local cost volume at the changed cells around the last match, GRU and regressor on the regions"""
        m, r = self.m, self.r
        f3 = feat[2]
        B, C, H, W = f3.shape
        dev = f3.device
        feat_cs = m.attend_full(f3, state['feat_cs'])
        self._t('attention (full)')
        *cl, fm1, fm2 = m.cnet(feat_cs)
        fmap = torch.cat([fm1, fm2], 0).float()
        zq = m.zqr[0](torch.relu(cl[0][1]))
        ctx = list(zq.split(zq.shape[1] // 3, 1))
        self._t('cnet (full)')

        # local cost volume at the changed positions, window centred on the last match
        pstar = state['coords1'][:, 0]
        b, y, x = M[:, 0].nonzero(as_tuple=True)
        cv, centers = state['cv'].clone(), state['centers'].clone()
        win = pstar[b, y, x][:, None] + torch.arange(-r, r + 1, device=dev).float()
        cv[b, :, y, x] = (sample_row_pts(fmap, 1 - b, y, win) * fmap[b, :, y, x][:, None]).sum(-1) / math.sqrt(fmap.shape[1])
        centers[b, 0, y, x] = pstar[b, y, x]
        self._t('local cost volume')

        boxes = region_boxes(M, self.margin)
        sls = [((slice(b, b + 1), slice(None), slice(y0, y1), slice(x0, x1)),
                (slice(b, b + 1), slice(None), slice(y0 * CELL, y1 * CELL), slice(x0 * CELL, x1 * CELL)), b)
               for b, y0, y1, x0, x1 in boxes]
        self._t('regions')

        net, coords1, coords0 = state['net'].clone(), state['coords1'].clone(), state['coords0']
        Mf = F.interpolate(M.float(), scale_factor=CELL, mode='nearest') > 0.5
        flow = state['flow'].clone()
        for sl, slf, bv in sls:
            net[sl], coords1[sl], up = self.gru(net[sl], [c[sl] for c in ctx], cv[sl], centers[sl], coords1[sl], coords0[sl], M[sl])
            flow[slf] = torch.where(Mf[slf], up, flow[slf])
        self._t('GRU + upsample')

        g = {k: v.clone() for k, v in state['g'].items()}
        xyz = g['xyz'].view(B, Mf.shape[-2], Mf.shape[-1], 3)
        for (sl, slf, _), (bv, y0, y1, x0, x1) in zip(sls, boxes):
            cam = {k: inp[k][bv:bv + 1] for k in ('intr', 'ref_intr', 'tf_x', 'extr')}
            dep = m.to_depth(flow[slf], cam)
            s2 = (slice(bv, bv + 1), slice(None), slice(y0 * 2, y1 * 2), slice(x0 * 2, x1 * 2))
            s4 = (slice(bv, bv + 1), slice(None), slice(y0 * 4, y1 * 4), slice(x0 * 4, x1 * 4))
            rot, scale, opacity, res = self.regress(inp['image'][slf], dep, feat[0][s4], feat[1][s2], feat_cs[sl], bv)
            mf = Mf[slf]
            depth = dep + res
            g['depth'][slf] = torch.where(mf, depth, g['depth'][slf])
            g['rot'][slf] = torch.where(mf, rot, g['rot'][slf])
            g['scale'][slf] = torch.where(mf, scale, g['scale'][slf])
            g['opacity'][slf] = torch.where(mf, opacity, g['opacity'][slf])
            g['img'][slf] = torch.where(mf, inp['image'][slf], g['img'][slf])
            K, E = cam['intr'][0], cam['extr'][0]
            yy = torch.arange(y0 * CELL, y1 * CELL, device=dev, dtype=torch.float32)[:, None] + 0.5
            xx = torch.arange(x0 * CELL, x1 * CELL, device=dev, dtype=torch.float32)[None, :] + 0.5
            z = 1.0 / (depth[0, 0] + 1e-8)
            p = torch.stack([(xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z], -1)
            xyz[bv, y0 * CELL:y1 * CELL, x0 * CELL:x1 * CELL] = torch.where(mf[0, 0, ..., None], (p - E[:, 3]) @ E[:, :3],
                                                                         xyz[bv, y0 * CELL:y1 * CELL, x0 * CELL:x1 * CELL])
        FrozenNorm.view_ids = None
        self._t('Gaussian regressor')

        st.update({'f3': torch.where(M, f3, state['f3']), 'feat_cs': feat_cs, 'centers': centers, 'cv': cv, 'net': net,
                   'coords1': coords1, 'flow': flow, 'g': g, 'image': torch.where(Mf, inp['image'], state['image']),
                   'fmap': fmap, 'ctx': ctx})
        area = sum((y1 - y0) * (x1 - x0) for _, y0, y1, x0, x1 in boxes) / (B * H * W)
        out = {'g': g, 'mask': M.float(), 'changed': changed.float(), 'p_star': pstar[:, None], 'flow': flow,
               'n_cells': int(M.sum()), 'n_boxes': len(boxes), 'area': area}
        return out, st

    # ---------------- frame t ----------------
    @torch.no_grad()
    def step(self, data, state, mask=None, use_proposal=True):
        m, r = self.m, self.r
        if self.timing is not None:
            torch.cuda.synchronize()
            self._last = time.perf_counter()
        inp = m.inputs(data)
        feat = self.encode(inp['image'])
        f3 = feat[2]
        self._t('encoder')
        logits = self.stss_logits(f3, state['f3'])
        changed = (torch.sigmoid(logits) > 0.5) if mask is None else mask > 0.5
        M = dilate(changed, m.update_dilation)  # cells that get new values
        self._t('STSS')
        B, C, H, W = f3.shape
        dev = f3.device
        st = {k: v for k, v in state.items()}
        if not M.any():  # nothing updated: the STSS reference stays the frame the kept values come from
            return {'g': state['g'], 'mask': M.float(), 'changed': changed.float(), 'n_cells': 0}, st
        if getattr(m, 'full_attention', False):
            return self._step_full(inp, feat, state, st, M, changed)

        # ---- proposal + local epipolar attention (point-wise, on the positions that need it) ----
        c_prev = state['centers'][:, 0]
        pstar = c_prev.clone()
        done = torch.zeros(B, H, W, dtype=torch.bool, device=dev)

        def ensure_pstar(sel):
            new = sel & ~done
            if not use_proposal or not new.any():
                done[new] = True
                return
            b, y, x = new.nonzero(as_tuple=True)
            fo = sample_row_pts(f3, 1 - b, y, c_prev[b, y, x][:, None])[:, 0]
            z = torch.cat([f3[b, :, y, x], state['f3'][b, :, y, x], fo, ((c_prev[b, y, x] - x.float()) / 32)[:, None]], 1)
            pstar[b, y, x] = c_prev[b, y, x] + mlp_pts(m.proposal.mlp, z)[:, 0] * 8
            done[new] = True

        def keys_of(sel):
            """positions of the other view read by the windows p* +- r of the queries sel"""
            b, y, x = sel.nonzero(as_tuple=True)
            xs = (torch.floor(pstar[b, y, x])[:, None] + torch.arange(-r, r + 2, device=dev)).clamp(0, W - 1).long()
            k = torch.zeros(B, H, W, dtype=torch.bool, device=dev)
            k[(1 - b)[:, None].expand_as(xs), y[:, None].expand_as(xs), xs] = True
            return k

        if self.patch:  # every cell of the patches is input to the residual block
            from lib.temporal_patch import patch_tiles, patch_cells, patch_convs
            idx = patch_tiles(M, self.patch[0])
            Cp = patch_cells(M, idx, *self.patch)[:, 0]
        else:
            Cp = dilate(M, 2)[:, 0]  # the residual block after the attention reads +-2 cells
        view = torch.arange(B, device=dev).view(B, 1, 1)
        Q4 = Cp & (view == 1)
        ensure_pstar(Q4)
        Q3 = (Cp & (view == 0)) | keys_of(Q4)
        ensure_pstar(Q3)
        Q2 = (Cp & (view == 1)) | keys_of(Q3)
        ensure_pstar(Q2)
        Q1 = Q3 | keys_of(Q2)
        ensure_pstar(Q1)
        self._t('proposal MLP')

        def attend(layer, x, src, sel):
            out = x.clone()
            b, y, xx = sel.nonzero(as_tuple=True)
            xs = pstar[b, y, xx][:, None] + torch.arange(-r, r + 1, device=dev).float()
            out[b, :, y, xx] = local_attention_pts(layer, x[b, :, y, xx], sample_row_pts(src, 1 - b, y, xs))
            return out
        L1, L2 = m.att_layers
        f_1 = attend(L1, f3, f3, Q1 & (view == 0))               # left after layer 1
        f_1 = attend(L1, f_1, f_1, Q2 & (view == 1))             # right after layer 1 (keys: left after layer 1)
        f_2 = attend(L2, f_1, f_1, Q3 & (view == 0))             # left after layer 2
        f_2 = attend(L2, f_2, f_2, Q4 & (view == 1))             # right after layer 2
        self._t('local attention')

        if self.patch:
            n = idx.shape[0]
            res = patch_convs(m, M, idx, *self.patch, f_2, pstar[:, None], feat, inp, state, pad_to=-(-n // self.bucket) * self.bucket)
            self._t('patches')
            Mf = res.pop('Mf')
            st.update(res)
            st.update({'f3': torch.where(M, f3, state['f3']), 'image': torch.where(Mf, inp['image'], state['image'])})
            out = {'g': res['g'], 'mask': M.float(), 'changed': changed.float(), 'p_star': pstar[:, None], 'flow': res['flow'],
                   'n_cells': int(M.sum()), 'n_boxes': n, 'area': n * (self.patch[0] + 2 * self.patch[1]) ** 2 / (B * H * W)}
            return out, st

        # regions: connected parts of (changed region + margin) as bounding boxes in cells (one host sync)
        boxes = region_boxes(M, self.margin)
        sls = [((slice(b, b + 1), slice(None), slice(y0, y1), slice(x0, x1)),
                (slice(b, b + 1), slice(None), slice(y0 * CELL, y1 * CELL), slice(x0 * CELL, x1 * CELL)), b)
               for b, y0, y1, x0, x1 in boxes]
        self._t('regions')

        # attention residual block (reads +-2 cells): new attention features at the changed cells
        feat_cs = state['feat_cs'].clone()
        for sl, _, bv in sls:
            feat_cs[sl] = torch.where(M[sl], self.resid(f_2[sl], bv), feat_cs[sl])
        self._t('attention resblock')

        # cnet where its input changed: feature head (+-3 cells), context + zqr conv (+-4 cells)
        Rc = dilate(M, 4)
        fmap, ctx = state['fmap'].clone(), [c.clone() for c in state['ctx']]
        for sl, _, bv in sls:
            fm, zs = self.cnet(feat_cs[sl], bv)
            fmap[sl] = torch.where(Rc[sl], fm, fmap[sl])
            for i, z in enumerate(zs):
                ctx[i][sl] = torch.where(Rc[sl], z, ctx[i][sl])
        self._t('cnet')

        # local cost volume at the changed positions
        b, y, x = M[:, 0].nonzero(as_tuple=True)
        cv, centers = state['cv'].clone(), state['centers'].clone()
        win = pstar[b, y, x][:, None] + torch.arange(-r, r + 1, device=dev).float()
        cv[b, :, y, x] = (sample_row_pts(fmap, 1 - b, y, win) * fmap[b, :, y, x][:, None]).sum(-1) / math.sqrt(fmap.shape[1])
        centers[b, 0, y, x] = pstar[b, y, x]
        self._t('local cost volume')

        # GRU (all iterations) + convex upsampling on the regions, updates the changed cells
        net, coords1, coords0 = state['net'].clone(), state['coords1'].clone(), state['coords0']
        Mf = F.interpolate(M.float(), scale_factor=CELL, mode='nearest') > 0.5
        flow = state['flow'].clone()
        for sl, slf, bv in sls:
            net[sl], coords1[sl], up = self.gru(net[sl], [c[sl] for c in ctx], cv[sl], centers[sl], coords1[sl], coords0[sl], M[sl])
            flow[slf] = torch.where(Mf[slf], up, flow[slf])
        self._t('GRU + upsample')

        # Gaussian regressor on the regions (full resolution), new Gaussians at the changed pixels
        g = {k: v.clone() for k, v in state['g'].items()}
        xyz = g['xyz'].view(B, Mf.shape[-2], Mf.shape[-1], 3)
        for (sl, slf, _), (bv, y0, y1, x0, x1) in zip(sls, boxes):
            cam = {k: inp[k][bv:bv + 1] for k in ('intr', 'ref_intr', 'tf_x', 'extr')}
            dep = m.to_depth(flow[slf], cam)
            s2 = (slice(bv, bv + 1), slice(None), slice(y0 * 2, y1 * 2), slice(x0 * 2, x1 * 2))
            s4 = (slice(bv, bv + 1), slice(None), slice(y0 * 4, y1 * 4), slice(x0 * 4, x1 * 4))
            rot, scale, opacity, res = self.regress(inp['image'][slf], dep, feat[0][s4], feat[1][s2], feat_cs[sl], bv)
            mf = Mf[slf]
            depth = dep + res
            g['depth'][slf] = torch.where(mf, depth, g['depth'][slf])
            g['rot'][slf] = torch.where(mf, rot, g['rot'][slf])
            g['scale'][slf] = torch.where(mf, scale, g['scale'][slf])
            g['opacity'][slf] = torch.where(mf, opacity, g['opacity'][slf])
            g['img'][slf] = torch.where(mf, inp['image'][slf], g['img'][slf])
            # lib.utils.depth2pc on the box
            K, E = cam['intr'][0], cam['extr'][0]
            yy = torch.arange(y0 * CELL, y1 * CELL, device=dev, dtype=torch.float32)[:, None] + 0.5
            xx = torch.arange(x0 * CELL, x1 * CELL, device=dev, dtype=torch.float32)[None, :] + 0.5
            z = 1.0 / (depth[0, 0] + 1e-8)
            p = torch.stack([(xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z], -1)
            xyz[bv, y0 * CELL:y1 * CELL, x0 * CELL:x1 * CELL] = torch.where(mf[0, 0, ..., None], (p - E[:, 3]) @ E[:, :3],
                                                                         xyz[bv, y0 * CELL:y1 * CELL, x0 * CELL:x1 * CELL])
        FrozenNorm.view_ids = None
        self._t('Gaussian regressor')

        Mf = F.interpolate(M.float(), scale_factor=CELL, mode='nearest') > 0.5
        st.update({'f3': torch.where(M, f3, state['f3']), 'feat_cs': feat_cs, 'centers': centers, 'cv': cv, 'net': net,
                   'coords1': coords1, 'flow': flow, 'g': g, 'image': torch.where(Mf, inp['image'], state['image']),
                   'fmap': fmap, 'ctx': ctx})
        area = sum((y1 - y0) * (x1 - x0) for _, y0, y1, x0, x1 in boxes) / (B * H * W)
        out = {'g': g, 'mask': M.float(), 'changed': changed.float(), 'p_star': pstar[:, None], 'flow': flow,
               'n_cells': int(M.sum()), 'n_boxes': len(boxes), 'area': area}
        return out, st
