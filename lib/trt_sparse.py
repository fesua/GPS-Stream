import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from lib.temporal_sparse import FrozenNorm
from lib.frozen_norm import FrozenGroupNorm
from export_trt import LOG, explicit_group_norm

# TensorRT version of the tile-cell execution (lib/patch_sparse.py), stage by stage:
#   dense engines (batch = both views, static shapes): front = encoder + epipolar attention + attention residual block +
#   context net; gru = cost volume + GRU x iters + convex upsampling; regr = Gaussian regressor;
#   crop engines (batch N = number of tile crops, dynamic): the same three stages on tile + halo crops (the GRU crop's cost
#   volume is the full row of the other view, as in PatchSparse). Normalisation layers of the crop stages are the
#   statistics recorded on the start frame (FrozenNorm / FrozenGroupNorm), averaged over the two views, as fixed affines.
# Written with -1 reshapes so that the ONNX graph keeps the batch dimension dynamic.


class FixedAffine(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer('scale', scale.detach().clone()); self.register_buffer('offset', offset.detach().clone())

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1) + self.offset.view(1, -1, 1, 1)


class LiveNorm(nn.Module):
    """group / instance norm with live statistics as plain reductions (the opset-17 export of GroupNorm is wrong in TRT)"""
    def __init__(self, groups, eps, weight, bias):
        super().__init__()
        self.g, self.eps, self.weight, self.bias = groups, eps, weight, bias

    def forward(self, x):
        N, C, H, W = x.shape
        xg = x.reshape(-1, self.g, (C // self.g) * H * W)
        mean = xg.mean(-1, keepdim=True); var = ((xg - mean) ** 2).mean(-1, keepdim=True)
        y = ((xg - mean) / torch.sqrt(var + self.eps)).reshape(-1, C, H, W)
        return y * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1) if self.weight is not None else y


def _stats(fn):
    if isinstance(fn, FrozenNorm):
        G = fn.groups; C = fn.weight.numel() if fn.weight is not None else G
    else:
        G, C = fn.G, fn.C
    return G, C


def affine_from(fn):
    G, C = _stats(fn)
    assert fn.mean is not None, 'statistics not recorded (run the start frame first)'
    mean, var = fn.mean.mean(0)[:, 0].repeat_interleave(C // G), fn.var.mean(0)[:, 0].repeat_interleave(C // G)
    scale = torch.rsqrt(var + fn.eps) * (fn.weight if fn.weight is not None else 1.0)
    offset = (fn.bias if fn.bias is not None else 0.0) - mean * scale
    return FixedAffine(scale, offset)


def live_from(fn):
    G, C = _stats(fn)
    return LiveNorm(G, fn.eps, fn.weight, fn.bias)


def fold(module, frozen):
    """FrozenNorm / FrozenGroupNorm wrappers -> fixed affines (frozen=True, crop stages) or live reductions (dense stages);
    remaining nn.GroupNorm -> explicit reductions"""
    for name, child in list(module.named_children()):
        if isinstance(child, (FrozenNorm, FrozenGroupNorm)):
            setattr(module, name, affine_from(child) if frozen else live_from(child))
        else:
            fold(child, frozen)
    return explicit_group_norm(module)


# ---------------- stages ----------------
def corr_pyramid(fm1, fm2, levels):
    """fm1 (N, D, h, w1), fm2 (N, D, h, w2) -> [(N h w1, 1, 1, w2 / 2^i)]"""
    D, w2 = fm1.shape[1], fm2.shape[-1]
    corr = torch.einsum('aijk,aijh->ajkh', fm1, fm2) / math.sqrt(D)
    corr = corr.reshape(-1, 1, 1, w2)
    pyr = [corr]
    for _ in range(levels):
        corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2]); pyr.append(corr)
    return pyr


class CorrLookup(nn.Module):
    def __init__(self, radius, levels, h, w1):
        super().__init__()
        self.r, self.levels, self.h, self.w1 = radius, levels, h, w1
        self.register_buffer('dx', torch.linspace(-radius, radius, 2 * radius + 1).view(2 * radius + 1, 1))

    def forward(self, pyr, cx):
        """cx (N, 1, h, w1) absolute x coordinate in the other view's row -> (N, levels (2r+1), h, w1)"""
        c = cx.reshape(-1, 1, 1, 1)
        outs = []
        for i in range(self.levels):
            corr = pyr[i]; W = corr.shape[-1]
            x0 = self.dx + c / 2 ** i
            grid = torch.cat([2 * x0 / (W - 1) - 1, torch.zeros_like(x0)], -1)          # (M, 2r+1, 1, 2)
            s = F.grid_sample(corr, grid, align_corners=True)                           # (M, 1, 2r+1, 1)
            outs.append(s.reshape(-1, self.h, self.w1, 2 * self.r + 1))
        return torch.cat(outs, -1).permute(0, 3, 1, 2)


class ConvexUp(nn.Module):
    def __init__(self, f=8):
        super().__init__()
        self.f = f
        self.register_buffer('k', torch.eye(9).view(9, 1, 3, 3))

    def forward(self, flow, mask):
        """flow (N, 1, h, w), mask (N, 9 f f, h, w) -> (N, 1, f h, f w); same arithmetic as RAFT's unfold-based upsampling"""
        f = self.f; h, w = flow.shape[-2:]
        m = torch.softmax(mask.reshape(-1, 1, 9, f, f, h, w), 2)
        up = F.conv2d(f * flow, self.k, padding=1).reshape(-1, 1, 9, 1, 1, h, w)
        up = (m * up).sum(2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(-1, 1, f * h, f * w)


class GRUStage(nn.Module):
    """cost volume + GRU x iters + upsampling on (N, D, h, w) features against (N, D, h, W) rows of the other view"""
    def __init__(self, rs, iters, h, w):
        super().__init__()
        self.upd, self.iters = rs.update_module.update_block, iters
        self.lookup = CorrLookup(rs.args.corr_radius, rs.args.corr_levels, h, w)
        self.levels = rs.args.corr_levels
        self.up = ConvexUp(2 ** rs.args.n_downsample)

    def forward(self, fm, rows, net, ctx, c0x, fi):
        pyr = corr_pyramid(fm, rows, self.levels)
        ctx3 = list(ctx.split(ctx.shape[1] // 3, 1))
        cx = c0x + fi
        for _ in range(self.iters):
            corr = self.lookup(pyr, cx)
            flow2 = torch.cat([cx - c0x, torch.zeros_like(cx)], 1)
            nl, up_mask, delta = self.upd([net], [ctx3], corr, flow2, iter32=False, iter16=False)
            net = nl[0]; cx = cx + delta[:, :1]
        return self.up(cx - c0x, up_mask)


class Front(nn.Module):
    """dense: image pair -> encoder features, attention outputs, attended features, fmap, GRU start state, context"""
    def __init__(self, t):
        super().__init__()
        self.enc, self.lo, self.cnet, self.zqr = t.img_encoder, t.loftr_coarse, t.raft_stereo.cnet, t.raft_stereo.context_zqr_convs[0]

    def tail(self, fcs):
        fm = self.cnet.conv2(fcs)
        net = torch.tanh(self.cnet.outputs08[0](fcs))
        ctx = self.zqr(torch.relu(self.cnet.outputs08[1](fcs)))
        return fm, net, ctx

    def forward(self, image):
        f1, f2, f3 = self.enc(image)
        L = self.lo.layers
        a0, a1 = f3[:1], f3[1:]
        a0 = L[0](a0, a1); a1 = L[0](a1, a0); att1 = torch.cat([a0, a1])
        a0 = L[1](a0, a1); a1 = L[1](a1, a0); att2 = torch.cat([a0, a1])
        fcs = self.lo.resid(att2)
        fm, net, ctx = self.tail(fcs)
        return f1, f2, f3, att1, att2, fcs, fm, net, ctx


class CropFront(Front):
    """crops: attention output crop -> attended features, fmap, GRU start state, context (frozen statistics)"""
    def forward(self, att2):
        fcs = self.lo.resid(att2)
        fm, net, ctx = self.tail(fcs)
        return fcs, fm, net, ctx


class DenseGRU(nn.Module):
    def __init__(self, rs, iters, h=128, w=128):
        super().__init__()
        self.g = GRUStage(rs, iters, h, w)
        self.register_buffer('c0x', torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(2, 1, h, w).contiguous())
        self.register_buffer('swap', torch.tensor([1, 0]))

    def forward(self, fm, net, ctx, fi):
        return self.g(fm, fm.index_select(0, self.swap), net, ctx, self.c0x, fi)


class Regr(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.reg = t.gs_parm_regresser

    def forward(self, image, depth, f1, f2, fcs):
        return self.reg(image, depth, (f1, f2, fcs))


# ---------------- TensorRT ----------------
def build_engine(onnx_path, engine_path, fp16, dyn=None, workspace_gb=8):
    """dyn: {input name: (min, opt, max) shapes} for a dynamic batch"""
    builder = trt.Builder(LOG)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOG)
    assert parser.parse_from_file(onnx_path), [parser.get_error(i) for i in range(parser.num_errors)]
    bc = builder.create_builder_config()
    bc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        bc.set_flag(trt.BuilderFlag.FP16)
    if dyn:
        prof = builder.create_optimization_profile()
        for n, (mn, op, mx) in dyn.items():
            prof.set_shape(n, mn, op, mx)
        bc.add_optimization_profile(prof)
    ser = builder.build_serialized_network(network, bc)
    assert ser is not None, 'engine build failed: ' + engine_path
    open(engine_path, 'wb').write(ser)


class TrtNet:
    """runs an engine on torch CUDA tensors; dynamic input shapes are set per call, outputs are allocated per shape"""
    def __init__(self, engine_path, inputs=None, outputs=None, graph=False, pad=8):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(open(engine_path, 'rb').read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = inputs or [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]      # order of the call arguments
        self.outputs = outputs or [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]  # order of the returned list
        self.dynamic = any(-1 in tuple(self.engine.get_tensor_shape(n)) for n in self.inputs)
        # graph=True: the engine launch is captured once per batch size (padded to a multiple of `pad`) into a CUDA graph and replayed
        # with static I/O buffers. For the ring engines (hundreds of small slice / concat layers per frame) the kernel launches, not
        # the computation, are the cost; the computation is unchanged (padded samples are zeros and are dropped).
        self.graph, self.pad, self.graphs = graph, pad, {}

    def _graph_call(self, args):
        n = args[0].shape[0]; npd = -(-n // self.pad) * self.pad
        key = (npd,) + tuple(tuple(t.shape[1:]) for t in args)
        if key not in self.graphs:
            ins = {nm: torch.zeros((npd,) + tuple(t.shape[1:]), device='cuda', dtype=t.dtype) for nm, t in zip(self.inputs, args)}
            for nm, t in ins.items():
                if self.dynamic:
                    self.ctx.set_input_shape(nm, tuple(t.shape))
                self.ctx.set_tensor_address(nm, t.data_ptr())
            outs = {nm: torch.zeros(tuple(self.ctx.get_tensor_shape(nm)), device='cuda') for nm in self.outputs}
            for nm, t in outs.items():
                self.ctx.set_tensor_address(nm, t.data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()   # warm-up launch with these shapes before the capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            self.graphs[key] = (g, ins, outs)
        g, ins, outs = self.graphs[key]
        for nm, t in zip(self.inputs, args):
            ins[nm][:n].copy_(t)
        g.replay()
        return [outs[nm][:n].clone() for nm in self.outputs]

    def __call__(self, *args):
        if self.graph:
            return self._graph_call(args)
        key, held = [], []
        for n, t in zip(self.inputs, args):
            t = t.contiguous(); held.append(t)   # keep every contiguous copy alive until the engine has run (a freed copy's memory is reused at once)
            if self.dynamic:
                self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr()); key.append(tuple(t.shape))
        # outputs are allocated per call (the caching allocator makes this cheap): a per-shape cache grew without bound with the
        # ring engines, whose outputs include one tile input per layer and whose batch size differs from frame to frame
        outs = {n: torch.empty(tuple(self.ctx.get_tensor_shape(n)), device='cuda') for n in self.outputs}
        for n, t in outs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        del held   # after the launch on the same stream
        return [outs[n] for n in self.outputs]

    def call_named(self, **inputs):
        """inputs by name; names the engine does not have (pruned unused inputs) are ignored; returns {output name: tensor}"""
        present = [n for n in self.inputs if n in inputs]
        assert len(present) == len(self.inputs), 'missing engine inputs: %s' % [n for n in self.inputs if n not in inputs]
        outs = self(*[inputs[n] for n in self.inputs])
        return dict(zip(self.outputs, outs))


# ---------------- per-sample normalisation statistics as an engine INPUT ----------------
# The crop stages normalise with the statistics of the sequence's start frame (per view). Those differ per sequence, so
# they cannot be engine constants: each crop engine takes `aff` (N, P) = [scale | offset] of every frozen norm layer for
# the view of each crop, and every layer slices its own channels. Layout = the unique frozen layers in named_modules order.
class AffCtx:
    aff = None   # (N, P) of the current call


class InputAffine(nn.Module):
    def __init__(self, off, C):
        super().__init__()
        self.off, self.C = off, C

    def forward(self, x):
        a = AffCtx.aff
        scale = a[:, self.off:self.off + self.C].reshape(-1, self.C, 1, 1)
        offset = a[:, self.off + self.C:self.off + 2 * self.C].reshape(-1, self.C, 1, 1)
        return x * scale + offset


def affine_layout(modules):
    """[(layer, off, C)] over the unique FrozenNorm / FrozenGroupNorm layers of the given modules (named_modules order,
    a layer referenced twice once) and the total length P"""
    out, off, seen = [], 0, set()
    for mod in modules:
        for _, m in mod.named_modules(remove_duplicate=False):
            if isinstance(m, (FrozenNorm, FrozenGroupNorm)) and id(m) not in seen:
                seen.add(id(m)); G, C = _stats(m); out.append((m, off, C)); off += 2 * C
    return out, off


def affine_vectors(layout, P):
    """(2, P): per view [scale | offset] of every layer from the recorded start-frame statistics"""
    vec = torch.zeros(2, P, device='cuda')
    for m, off, C in layout:
        G, _ = _stats(m)
        for v in range(2):
            mean, var = m.mean[v][:, 0].repeat_interleave(C // G), m.var[v][:, 0].repeat_interleave(C // G)
            scale = torch.rsqrt(var + m.eps) * (m.weight if m.weight is not None else 1.0)
            offset = (m.bias if m.bias is not None else 0.0) - mean * scale
            vec[v, off:off + C] = scale; vec[v, off + C:off + 2 * C] = offset
    return vec


def fold_input_affine(root, modules):
    """replace the frozen norm layers of `modules` (children of root) by InputAffine slices of AffCtx.aff; returns P"""
    layout, P = affine_layout(modules)
    repl = {id(m): InputAffine(off, C) for m, off, C in layout}
    def rec(module):
        for name, child in list(module.named_children()):
            if id(child) in repl:
                setattr(module, name, repl[id(child)])
            else:
                rec(child)
    for mod in modules:
        rec(mod)
    explicit_group_norm(root)
    return P


class CropFrontAff(CropFront):
    def forward(self, att2, aff):
        AffCtx.aff = aff
        return CropFront.forward(self, att2)


class RegrAff(Regr):
    def forward(self, image, depth, f1, f2, fcs, aff):
        AffCtx.aff = aff
        return self.reg(image, depth, (f1, f2, fcs))


class RowAttn(nn.Module):
    """one Crossatt_EncoderLayer on a batch of ROWS: x_rows, src_rows (R, W, C) -> (R, W, C); the same arithmetic as the
    dense layer and as att_sparse (per-row attention of every query of the row over the source row). Only the rows that
    contain tile cells are passed, and the caller keeps the tile cells of the output."""
    def __init__(self, layer):
        super().__init__()
        self.l = layer

    def forward(self, x_rows, src_rows):
        l, mha = self.l, self.l.attention
        q = l.q_proj(l.norm1(x_rows)); s = l.norm1(src_rows); k, v = l.k_proj(s), l.v_proj(s)
        # nn.MultiheadAttention written out with -1 reshapes: the module's own export bakes the batch size into its views, which
        # breaks a dynamic number of rows
        E, h = mha.embed_dim, mha.num_heads; d = E // h; W = x_rows.shape[1]
        wq, wk, wv = mha.in_proj_weight.chunk(3); bq, bk, bv = mha.in_proj_bias.chunk(3)
        Q = F.linear(q, wq, bq).reshape(-1, W, h, d).transpose(1, 2)
        K = F.linear(k, wk, bk).reshape(-1, W, h, d).transpose(1, 2)
        V = F.linear(v, wv, bv).reshape(-1, W, h, d).transpose(1, 2)
        A = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d), -1)
        m = torch.matmul(A, V).transpose(1, 2).reshape(-1, W, E)
        m = l.merge(F.linear(m, mha.out_proj.weight, mha.out_proj.bias))
        return x_rows + l.norm2(l.mlp(torch.cat([x_rows, m], -1)))


# ---------------- stored-context decoding (variant D): engines with one ring input per convolution ----------------
class GRUStageRing(GRUStage):
    """GRU stage returning the low-res flow and the upsampling mask (the convex upsampling with the flow ring runs outside)"""
    def forward(self, fm, rows, net, ctx, c0x, fi):
        pyr = corr_pyramid(fm, rows, self.levels)
        ctx3 = list(ctx.split(ctx.shape[1] // 3, 1))
        cx = c0x + fi
        for _ in range(self.iters):
            corr = self.lookup(pyr, cx)
            flow2 = torch.cat([cx - c0x, torch.zeros_like(cx)], 1)
            nl, up_mask, delta = self.upd([net], [ctx3], corr, flow2, iter32=False, iter16=False)
            net = nl[0]; cx = cx + delta[:, :1]
        return cx - c0x, up_mask


class RingWrap(nn.Module):
    """runs a tile stage under TileCtx export mode: forward(*stage inputs, *rings) -> (*stage outputs, *tile inputs of the ring layers)"""
    def __init__(self, stage, n_in):
        super().__init__()
        self.stage, self.n_in = stage, n_in

    def forward(self, *args):
        from lib.tile_mode import TileCtx
        TileCtx.begin_export(args[self.n_in:])
        try:
            out = self.stage(*args[:self.n_in])
            outs = list(TileCtx.outs)
        finally:
            TileCtx.end()
        out = list(out) if isinstance(out, (tuple, list)) else [out]
        return tuple(out + outs)


class RingWrapG(nn.Module):
    """ring engine with GROUPED rings (lib/tile_mode.ring_groups): forward(*stage inputs, *group rings) -> (*stage outputs, *group tile
    inputs); every layer's ring is the channel slice of its group's ring (call order) and its tile input goes back into the same slice"""
    def __init__(self, stage, n_in, groups):
        super().__init__()
        self.stage, self.n_in, self.groups = stage, n_in, groups
        lay = {i: (j, off, C) for j, g in enumerate(groups) for i, off, C in g['layers']}
        self.slices = [lay[i] for i in range(len(lay))]

    def forward(self, *args):
        from lib.tile_mode import TileCtx
        rings = args[self.n_in:]
        TileCtx.begin_export([rings[j][:, off:off + C] for j, off, C in self.slices])
        try:
            out = self.stage(*args[:self.n_in])
            outs = list(TileCtx.outs)
        finally:
            TileCtx.end()
        assert len(outs) == len(self.slices), 'ring layers consumed %d, recorded %d' % (len(outs), len(self.slices))
        out = list(out) if isinstance(out, (tuple, list)) else [out]
        return tuple(out + [torch.cat([outs[i] for i, _, _ in g['layers']], 1) for g in self.groups])


class RingWrapB(nn.Module):
    """ring engine on band stores (lib/tile_mode.BandStore): forward(*stage inputs, tb_g0, lr_g0, tb_g1, lr_g1, ...) ->
    (*stage outputs, xtb_g0, xlr_g0, ...): per group the ring strips in, the fresh tiles' border strips out (channels = the
    group's layers in call order)"""
    def __init__(self, stage, n_in, groups):
        super().__init__()
        self.stage, self.n_in, self.groups = stage, n_in, groups
        lay = {i: (j, off, C) for j, g in enumerate(groups) for i, off, C in g['layers']}
        self.slices = [lay[i] for i in range(len(lay))]

    def forward(self, *args):
        from lib.tile_mode import TileCtx, tile_strips
        tb, lr = args[self.n_in::2], args[self.n_in + 1::2]
        TileCtx.begin_export([(tb[j][:, off:off + C], lr[j][:, off:off + C]) for j, off, C in self.slices])
        try:
            out = self.stage(*args[:self.n_in])
            outs = list(TileCtx.outs)
        finally:
            TileCtx.end()
        assert len(outs) == len(self.slices), 'ring layers consumed %d, recorded %d' % (len(outs), len(self.slices))
        out = list(out) if isinstance(out, (tuple, list)) else [out]
        strips = []
        for g in self.groups:
            st = [tile_strips(outs[i], g['r']) for i, _, _ in g['layers']]
            strips += [torch.cat([a for a, _ in st], 1), torch.cat([b for _, b in st], 1)]
        return tuple(out + strips)
