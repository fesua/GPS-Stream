import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.temporal_sparse import crop_tiles
from lib.temporal_patch import put_tiles

# "Stored-context decoding" (variant D): every convolution (and bilinear upsampling) of the tile stages computes ONLY the tile's
# outputs, and takes the neighbours it needs from a full-frame store of its own input, written the last time those cells were
# updated. The tile's fresh input replaces the centre of the gathered ring and is written back into the store. So the decoded
# range is the tile, the receptive field is not truncated, and the ring context is as old as its last update.
# Mechanics: nn.Conv2d.forward / nn.Upsample.forward are patched. mode 'record' (dense pass of the start frame) stores every
# layer input for the whole frame; mode 'tile' gathers rings from the stores. Layers are keyed by module id and call index
# within the frame (the GRU's convolutions run 8 times per frame).


class TileCtx:
    active = False
    mode = 'tile'      # 'record' | 'tile' | 'export'
    stores = None      # {(id, call): (2, C, H, W)}
    calls = None       # {id: call count in this frame}
    idx = None         # (n, 3) tiles: view, row, col (tile units)
    k8 = 16            # tile size in cells at 1/8
    w8 = 128           # width of the 1/8 map the stores were recorded at
    write = True
    meta = None        # record mode: {key: (padding r, store width)}; order: keys in call order; marks: {stage: start index in order}
    order = None
    marks = None
    rings = None       # export mode: iterator over the ring input tensors (call order)
    outs = None        # export mode: the tile inputs of every ring layer (call order), returned by the engine for the store write-back

    @classmethod
    def begin(cls, mode, stores, idx=None, k8=16, w8=128, write=True):
        cls.active, cls.mode, cls.stores, cls.calls, cls.idx, cls.k8, cls.w8, cls.write = True, mode, stores, {}, idx, k8, w8, write
        if mode == 'record':
            cls.meta, cls.order, cls.marks = {}, [], {}

    @classmethod
    def mark(cls, stage):
        if cls.active and cls.mode == 'record':
            cls.marks[stage] = len(cls.order)

    @classmethod
    def begin_export(cls, rings):
        cls.active, cls.mode, cls.calls, cls.rings, cls.outs = True, 'export', {}, iter(rings), []

    @classmethod
    def end(cls):
        cls.active = False; cls.calls = None


def ring_pad(x, ring, r):
    """(n, C, s, s) fresh tile into the centre of the gathered ring (n, C, s + 2r, s + 2r), without in-place writes (exportable)"""
    if r == 0:
        return x
    s = x.shape[-1]
    mid = torch.cat([ring[..., r:r + s, :r], x, ring[..., r:r + s, r + s:]], -1)
    return torch.cat([ring[..., :r, :], mid, ring[..., r + s:, :]], -2)


def ring_pad_strips(x, tb, lr, r):
    """like ring_pad with the ring given as strips: tb (n, C, 2r, s + 2r) = the r rows above then the r rows below the tile (full ring
    width), lr (n, C, s, 2r) = the r columns left then right of it"""
    if r == 0:
        return x
    mid = torch.cat([lr[..., :r], x, lr[..., r:]], -1)
    return torch.cat([tb[:, :, :r], mid, tb[:, :, r:]], -2)


def _key(mod):
    c = TileCtx.calls.get(id(mod), 0); TileCtx.calls[id(mod)] = c + 1
    return (id(mod), c)


def _tile_io(key, x, r):
    """x (n, C, s, s) tile crop at this layer's scale -> ring-padded crop (n, C, s + 2r, s + 2r) with the fresh tile in the centre;
    the store gets the fresh tile"""
    store = TileCtx.stores[key]
    s = x.shape[-1]
    ring = crop_tiles(store, TileCtx.idx, s, r)
    if TileCtx.write:
        put_tiles(store, x.detach() if not torch.is_grad_enabled() else x, TileCtx.idx, torch.ones(x.shape[0], 1, s, s, dtype=torch.bool, device=x.device))
    return ring_pad(x, ring, r)


def _layer_io(mod, x, r):
    """the ring-padded input of a patched layer in every mode; layers that need no neighbours (padding 0) are left alone"""
    key = _key(mod)
    if TileCtx.mode == 'record':
        TileCtx.stores[key] = x.detach().clone(); TileCtx.meta[key] = (r, x.shape[-1]); TileCtx.order.append(key)
        return None
    if TileCtx.mode == 'export':
        TileCtx.outs.append(x)
        rg = next(TileCtx.rings)
        return ring_pad_strips(x, rg[0], rg[1], r) if isinstance(rg, tuple) else ring_pad(x, rg, r)
    return _tile_io(key, x, r)


_conv_forward = nn.Conv2d.forward
_up_forward = nn.Upsample.forward


def _conv_tile(self, x):
    p = self.padding[0]
    if not TileCtx.active or p == 0:
        return _conv_forward(self, x)
    inp = _layer_io(self, x, p)
    if inp is None:
        return _conv_forward(self, x)
    return F.conv2d(inp, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def _up_tile(self, x):
    if not TileCtx.active:
        return _up_forward(self, x)
    s = x.shape[-1]; f = int(self.scale_factor)
    inp = _layer_io(self, x, 1)
    if inp is None:
        return _up_forward(self, x)
    y = _up_forward(self, inp)
    return y[..., f:f + f * s, f:f + f * s]


def install():
    nn.Conv2d.forward = _conv_tile
    nn.Upsample.forward = _up_tile


def convex_upsample_ring(flow_tile, mask, flow_store, f=8):
    """RAFT convex upsampling of the tile's low-res flow (n, D, k, k) with the 3x3 neighbourhood of the tile border from the
    full-frame low-res flow store (2, D, h, w); the store gets the fresh tile"""
    n, D, k, _ = flow_tile.shape
    ring = ring_pad(flow_tile, crop_tiles(flow_store, TileCtx.idx, k, 1), 1)
    if TileCtx.write:
        put_tiles(flow_store, flow_tile.detach() if not torch.is_grad_enabled() else flow_tile, TileCtx.idx, torch.ones(n, 1, k, k, dtype=torch.bool, device=flow_tile.device))
    m = torch.softmax(mask.view(n, 1, 9, f, f, k, k), dim=2)
    up = F.unfold(f * ring, [3, 3], padding=0).view(n, D, 9, 1, 1, k, k)
    up = torch.sum(m * up, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(n, D, f * k, f * k)


# ---------------- grouped rings (TensorRT engines of variant D) ----------------
# One gather / write-back per layer costs ~125 small index operations per frame. The ring layers of a stage are therefore grouped by
# (store width, ring r): one store per group with the layers' channels concatenated in call order and a border of r, so a group's
# rings are ONE strided gather (NCHW output, no padding copy) and its tile inputs ONE write-back. The computation is unchanged.
def ring_groups(stage_keys, meta, stores):
    """{stage: [{'sw': store width, 'r': ring, 'C': channels, 'layers': [(layer index in the stage, channel offset, C), ...]}]}"""
    out = {}
    for stage, keys in stage_keys.items():
        groups, by = [], {}
        for i, key in enumerate(keys):
            r, sw = meta[key]; C = stores[key].shape[1]
            g = by.get((sw, r))
            if g is None:
                g = {'sw': sw, 'r': r, 'C': 0, 'layers': []}; by[(sw, r)] = g; groups.append(g)
            g['layers'].append((i, g['C'], C)); g['C'] += C
        out[stage] = groups
    return out


def gather_rings(gs, idx, s, r):
    """padded group store gs (B, C, H + 2r, W + 2r) -> (n, C, s + 2r, s + 2r): the ring windows of the tiles idx (n, 3) [view, row, col]
    in one strided gather (the windows overlap by 2r, so they are a strided view, not a reshape)"""
    B, C, Hp, Wp = gs.shape; H, W = Hp - 2 * r, Wp - 2 * r
    win = gs.as_strided((B, H // s, W // s, C, s + 2 * r, s + 2 * r), (C * Hp * Wp, s * Wp, s, Hp * Wp, Wp, 1), gs.storage_offset())
    return win[idx[:, 0], idx[:, 1], idx[:, 2]]


def write_tiles_padded(gs, x, idx, r):
    """the fresh tile inputs x (n, C, s, s) into the interior of the padded group store gs at the tiles idx (one index write)"""
    B, C, Hp, Wp = gs.shape; s = x.shape[-1]; H, W = Hp - 2 * r, Wp - 2 * r
    win = gs.as_strided((B, H // s, W // s, C, s, s), (C * Hp * Wp, s * Wp, s, Hp * Wp, Wp, 1), gs.storage_offset() + r * Wp + r)
    win[idx[:, 0], idx[:, 1], idx[:, 2]] = x


class BandStore:
    """variant D stores reduced to what the ring gathers ever read: the rows and columns within r of the tile grid of a group's
    full-frame store (B, C, H, W). hb (B, C, (Hs + 2) * 2r, W + 2r): per tile row t (kept at t + 1, zero rows at both ends), its top
    r rows then its bottom r rows, columns padded by r; vb (B, C, H + 2r, (Ws + 2) * 2r): per tile column, its left r then its right
    r columns, rows padded by r. A tile's ring = the bottom rows of the tile row above + the top rows of the row below (hb, s + 2r
    wide: the corners come from the diagonal tiles' rows) and the right / left columns of the neighbouring tile columns (vb).
    One strided gather and one index write per store; the corner pixels live in both and are written in both."""
    def __init__(self, B, C, H, W, s, r, device='cuda'):
        self.B, self.C, self.H, self.W, self.s, self.r, self.Hs, self.Ws = B, C, H, W, s, r, H // s, W // s
        self.hb = torch.zeros(B, C, (self.Hs + 2) * 2 * r, W + 2 * r, device=device)
        self.vb = torch.zeros(B, C, H + 2 * r, (self.Ws + 2) * 2 * r, device=device)

    def fill(self, x, off):
        """the recorded full-frame input x (B, C_l, H, W) of one layer into channels off:off + C_l"""
        s, r, B, H, W = self.s, self.r, self.B, self.H, self.W; C_l = x.shape[1]
        hb = self.hb[:, off:off + C_l].view(B, C_l, self.Hs + 2, 2, r, W + 2 * r)
        vb = self.vb[:, off:off + C_l].view(B, C_l, H + 2 * r, self.Ws + 2, 2, r)
        xr = x.view(B, C_l, self.Hs, s, W); xc = x.view(B, C_l, H, self.Ws, s)
        hb[:, :, 1:-1, 0, :, r:r + W] = xr[:, :, :, :r]; hb[:, :, 1:-1, 1, :, r:r + W] = xr[:, :, :, s - r:]
        vb[:, :, r:r + H, 1:-1, 0] = xc[..., :r]; vb[:, :, r:r + H, 1:-1, 1] = xc[..., s - r:]

    def gather(self, idx):
        """-> tb (n, C, 2r, s + 2r), lr (n, C, s, 2r): the ring strips of the tiles idx (n, 3) [view, row, col]"""
        s, r, C, B = self.s, self.r, self.C, self.B; n = idx.shape[0]
        Wp, RH = self.W + 2 * r, self.hb.shape[2]
        tb = self.hb.as_strided((B, self.Hs, self.Ws, C, 2, r, s + 2 * r), (C * RH * Wp, 2 * r * Wp, s, RH * Wp, 3 * r * Wp, Wp, 1), self.hb.storage_offset() + r * Wp)
        Hp, RW = self.H + 2 * r, self.vb.shape[3]
        lr = self.vb.as_strided((B, self.Hs, self.Ws, C, s, 2, r), (C * Hp * RW, s * RW, 2 * r, Hp * RW, RW, 3 * r, 1), self.vb.storage_offset() + r * RW + r)
        return tb[idx[:, 0], idx[:, 1], idx[:, 2]].view(n, C, 2 * r, s + 2 * r), lr[idx[:, 0], idx[:, 1], idx[:, 2]].view(n, C, s, 2 * r)

    def write(self, idx, xtb, xlr):
        """the fresh tiles' own top / bottom r rows xtb (n, C, 2r, s) and left / right r columns xlr (n, C, s, 2r)"""
        s, r, C, B = self.s, self.r, self.C, self.B; n = idx.shape[0]
        Wp, RH = self.W + 2 * r, self.hb.shape[2]
        tb = self.hb.as_strided((B, self.Hs, self.Ws, C, 2, r, s), (C * RH * Wp, 2 * r * Wp, s, RH * Wp, r * Wp, Wp, 1), self.hb.storage_offset() + 2 * r * Wp + r)
        tb[idx[:, 0], idx[:, 1], idx[:, 2]] = xtb.view(n, C, 2, r, s)
        Hp, RW = self.H + 2 * r, self.vb.shape[3]
        lr = self.vb.as_strided((B, self.Hs, self.Ws, C, s, 2, r), (C * Hp * RW, s * RW, 2 * r, Hp * RW, RW, r, 1), self.vb.storage_offset() + r * RW + 2 * r)
        lr[idx[:, 0], idx[:, 1], idx[:, 2]] = xlr.view(n, C, s, 2, r)


def tile_strips(x, r):
    """(n, C, s, s) -> its top / bottom r rows (n, C, 2r, s) and left / right r columns (n, C, s, 2r): what the band stores keep"""
    return torch.cat([x[:, :, :r], x[:, :, -r:]], 2), torch.cat([x[..., :r], x[..., -r:]], 3)
