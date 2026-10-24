import torch
import torch.nn.functional as F

from lib.temporal_gps import lookup
from lib.temporal_sparse import FrozenNorm, crop_tiles, write_tiles, sample_row_pts
from lib.utils import depth2pc

# Patch execution of the conv parts of a TemporalGPS step (attention residual block, cnet, GRU + upsampling,
# Gaussian regressor): the feature map is split into fixed tiles of k x k cells; every tile holding an updated cell
# becomes one patch of (k + 2 halo) cells, and all patches go through each module as one batch. A patch is processed
# as a small image of its own (zero padding at its border); only the updated cells are written back.
# Training (TemporalGPS.step_patch) and inference (SparseTemporalGPS with patch=...) call the same function, so
# they compute the same thing. Normalisation layers use the statistics recorded at the keyframe (FrozenNorm).

CELL = 8


def put_tiles(dense, vals, idx, mask):
    """dense (B, C, H, W) <- vals (n, C, t, t) on the tiles idx (n, 3) where mask (n, 1, t, t); integer indexing
    only (no host synchronisation). H, W must be multiples of t"""
    B, C, H, W = dense.shape
    t = vals.shape[-1]
    v = dense.view(B, C, H // t, t, W // t, t)
    cur = v[idx[:, 0], :, idx[:, 1], :, idx[:, 2], :]  # (n, C, t, t)
    v[idx[:, 0], :, idx[:, 1], :, idx[:, 2], :] = torch.where(mask, vals, cur)


def patch_tiles(M, k):
    """(n, 3) = view, tile row, tile col of the k x k-cell tiles holding an updated cell"""
    return (F.max_pool2d(M.float(), k, k) > 0.5)[:, 0].nonzero()


def patch_cells(M, idx, k, halo):
    """(B, 1, H, W) bool: the cells covered by the patches (tile + halo)"""
    B, _, H, W = M.shape
    P = torch.zeros(B, 1, H + 2 * halo, W + 2 * halo, dtype=torch.bool, device=M.device)
    ar = torch.arange(k + 2 * halo, device=M.device)
    Y = (idx[:, 1] * k)[:, None, None] + ar[None, :, None]
    X = (idx[:, 2] * k)[:, None, None] + ar[None, None, :]
    P[idx[:, 0][:, None, None].expand_as(Y), 0, Y, X] = True
    return P[:, :, halo:halo + H, halo:halo + W]


def patch_groups(M, idx, k):
    """(B, 1, H * 8, W * 8) long per pixel: 0 kept, 1 + patch number for updated pixels (for the border loss)"""
    g = torch.zeros_like(M, dtype=torch.long)
    ids = torch.arange(1, idx.shape[0] + 1, device=M.device).view(-1, 1, 1, 1).expand(-1, 1, k, k)
    write_tiles(g, ids, idx, k, M)
    return F.interpolate(g.float(), scale_factor=CELL, mode='nearest').long()


def patch_convs(m, M, idx, k, halo, f2, pstar, feat, inp, state, pad_to=None):
    """m: TemporalGPS with FrozenNorm in att_resid, cnet, regressor. M (B,1,H,W) updated cells, idx from patch_tiles.
    f2: attention output (B,C,H,W), valid on the patch cells; pstar (B,1,H,W) valid on M.
    state: feat_cs, fmap, ctx (caches), cv, centers, net, coords1, coords0, flow, g.
    pad_to: pad the patch batch to this size with patches that write nothing (fixed shapes for cuDNN).
    Returns the new feat_cs, fmap, ctx, cv, centers, net, coords1, flow, g."""
    n = idx.shape[0]
    if pad_to is not None and pad_to > n:
        idx_run = torch.cat([idx, idx[:1].expand(pad_to - n, -1)], 0)
        live = torch.arange(pad_to, device=idx.device) < n
    else:
        idx_run, live = idx, torch.ones(n, dtype=torch.bool, device=idx.device)
    crop = lambda x, s=1: crop_tiles(x, idx_run, k * s, halo * s)
    core = lambda x, s=1: x[..., halo * s:(halo + k) * s, halo * s:(halo + k) * s]
    Mp = crop(M) & live.view(-1, 1, 1, 1)
    Mf = F.interpolate(M.float(), scale_factor=CELL, mode='nearest') > 0.5

    Mc = core(Mp)[:n]
    Mcf = F.interpolate(Mc.float(), scale_factor=CELL, mode='nearest') > 0.5

    def put(dense, vals, s=1):  # write the core of the live patches at the updated cells / pixels
        put_tiles(dense, core(vals, s)[:n], idx, Mc if s == 1 else Mcf)
    FrozenNorm.view_ids = idx_run[:, 0]

    # attention residual block -> new attention features at the updated cells
    fcs_p = torch.where(Mp, m.att_resid(crop(f2)), crop(state['feat_cs']))
    feat_cs = state['feat_cs'].clone()
    put(feat_cs, fcs_p)

    # cnet: feature head (cost volume) and context (GRU)
    fm_p = m.cnet.conv2(fcs_p)
    zq = m.zqr[0](torch.relu(m.cnet.outputs08[1](fcs_p)))
    ctx_p = list(zq.split(zq.shape[1] // 3, 1))
    fmap = state['fmap'].clone()
    put(fmap, fm_p)
    ctx = [c.clone() for c in state['ctx']]
    for c, cp in zip(ctx, ctx_p):
        put(c, cp)

    # local cost volume at the updated cells: p* +- r on the other view's feature map (point-wise, whole map)
    b, y, x = M[:, 0].nonzero(as_tuple=True)
    r = m.r
    win = pstar[b, 0, y, x][:, None] + torch.arange(-r, r + 1, device=M.device).float()
    cv, centers = state['cv'].clone(), state['centers'].clone()
    cv[b, :, y, x] = (sample_row_pts(fmap, 1 - b, y, win) * fmap[b, :, y, x][:, None]).sum(-1) / fmap.shape[1] ** 0.5
    centers[b, 0, y, x] = pstar[b, 0, y, x]

    # GRU iterations + convex upsampling on the patches, updates the updated cells only
    net_p, c1_p, c0_p = crop(state['net']), crop(state['coords1']), crop(state['coords0'])
    cv_p, ce_p = crop(cv), crop(centers)
    Mf_p = Mp.float()
    for _ in range(m.iters):
        corr = lookup(cv_p, ce_p, c1_p[:, :1], r)
        nl, up_mask, delta = m.update_block([net_p], [ctx_p], corr, c1_p - c0_p, iter32=False, iter16=False)
        net_p = torch.where(Mp, nl[0], net_p)
        c1_p = c1_p + torch.cat([delta[:, :1], torch.zeros_like(delta[:, 1:])], 1) * Mf_p
    up = m.upsample(c1_p - c0_p, up_mask)[:, :1]
    net, coords1 = state['net'].clone(), state['coords1'].clone()
    put(net, net_p)
    put(coords1, c1_p)
    flow = state['flow'].clone()
    put(flow, up, CELL)

    # Gaussian regressor on the patches (full resolution), new Gaussians at the updated pixels
    dep_full = m.to_depth(flow, inp)
    img_p, dep_p = crop(inp['image'], CELL), crop(dep_full, CELL)
    rot, scale, opacity, res = m.regressor(img_p, dep_p, (crop(feat[0], 4), crop(feat[1], 2), fcs_p))
    gp = state['g']
    g = {kk: v.clone() for kk, v in gp.items() if kk not in ('xyz', 'valid')}
    put(g['depth'], dep_p + res, CELL)
    put(g['rot'], rot, CELL)
    put(g['scale'], scale, CELL)
    put(g['opacity'], opacity, CELL)
    g['img'] = torch.where(Mf, inp['image'], gp['img'])
    g['xyz'] = depth2pc(g['depth'], inp['extr'], inp['intr'])
    g['valid'] = (inp['mask'][:, :1] > 0.5).flatten(1)
    FrozenNorm.view_ids = None
    return {'feat_cs': feat_cs, 'fmap': fmap, 'ctx': ctx, 'cv': cv, 'centers': centers, 'net': net, 'coords1': coords1,
            'flow': flow, 'g': g, 'Mf': Mf}
