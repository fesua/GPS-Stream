import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import CorrBlock1D
from core.utils.utils import downflow8, coords_grid
from lib.utils import depth2pc

# Temporal GPS-Gaussian+: frame t reuses frame t-1 wherever STSS finds no change.
#   keyframe : the per-frame model (teacher) computes everything and fills the state
#   frame t  : STSS(F_t, F_{t-1}) -> changed mask M (feature-map resolution, dilated by one cell)
#              changed cells: MLP proposal p* on the epipolar line -> local epipolar attention on p* +- R
#                             -> local cost volume (2R+1 values around p*) -> GRU update -> new Gaussians
#              unchanged    : attention features, cost volume, GRU state, flow and Gaussians of frame t-1
# Everything is computed on the full feature map and the mask selects new vs. reused values, so the
# outputs are those of the method while the network itself is unchanged elsewhere.

R = 4  # default local window radius on the epipolar line (feature-map cells)


def swap_views(x):
    # batch = [left, right] -> the other view of each sample
    return torch.cat([x[1:2], x[0:1]], 0)


def sample_row(feat, xs):
    """feat (B, C, H, W), xs (B, K, H, W) x-positions on the same row -> (B, C, K, H, W), zeros outside"""
    B, C, H, W = feat.shape
    K = xs.shape[1]
    gx = 2 * xs / (W - 1) - 1
    gy = torch.linspace(-1, 1, H, device=feat.device).view(1, 1, H, 1).expand(B, K, H, W)
    grid = torch.stack([gx, gy], -1).permute(0, 2, 1, 3, 4).reshape(B, H, K * W, 2)
    out = F.grid_sample(feat, grid, align_corners=True)  # B, C, H, K*W
    return out.view(B, C, H, K, W).permute(0, 1, 3, 2, 4)


def window_offsets(device, r=R):
    return torch.arange(-r, r + 1, dtype=torch.float32, device=device).view(1, -1, 1, 1)


def local_cost_volume(fmap, fmap_other, centers, r=R):
    """(2r+1) correlations of fmap(x) with fmap_other at centers(x) + k on the epipolar line"""
    win = sample_row(fmap_other, centers + window_offsets(fmap.device, r))  # B, D, 2r+1, H, W
    return (win * fmap.unsqueeze(2)).sum(1) / math.sqrt(fmap.shape[1])


def lookup(cv, centers, x, r=R):
    """GRU lookup: values of the local cost volume at x + j, j = -r..r, linear interpolation, zero outside"""
    pos = x + window_offsets(cv.device, r) - centers + r  # index into the window
    i0 = torch.floor(pos)
    w1 = pos - i0
    out = 0
    for i, w in [(i0, 1 - w1), (i0 + 1, w1)]:
        valid = (i >= 0) & (i <= 2 * r)
        v = torch.gather(cv, 1, i.clamp(0, 2 * r).long())
        out = out + v * w * valid
    return out


class STSS(nn.Module):
    """spatio-temporal self-similarity of F_t against a 5x5 neighbourhood of the reference features (per cell, the
    frame its kept values come from) -> change logits"""

    def __init__(self, dim=96, k=5):
        super().__init__()
        self.k = k
        self.head = nn.Sequential(
            nn.Conv2d(k * k + 1, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

    def forward(self, f, f_prev):
        B, C, H, W = f.shape
        fn, pn = F.normalize(f, dim=1), F.normalize(f_prev, dim=1)
        nb = F.unfold(pn, self.k, padding=self.k // 2).view(B, C, self.k * self.k, H, W)
        sim = (fn.unsqueeze(2) * nb).sum(1)
        diff = (f - f_prev).abs().mean(1, keepdim=True)
        return self.head(torch.cat([sim, diff], 1))


class ProposalMLP(nn.Module):
    """per-cell MLP: correspondence position p* on the epipolar line = previous correspondence + offset"""

    def __init__(self, dim=96, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(3 * dim + 1, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, f, f_prev, f_other_at_prev, disp_prev):
        return self.mlp(torch.cat([f, f_prev, f_other_at_prev, disp_prev / 32], 1)) * 8


def local_cross_attention(layer, x, source, centers, r=R):
    """lib.attention_module.Crossatt_EncoderLayer with keys restricted to centers +- r on the epipolar line"""
    B, C, H, W = x.shape
    att = layer.attention
    nh, hd = att.num_heads, C // att.num_heads
    q = layer.q_proj(layer.norm1(x.permute(0, 2, 3, 1)))                                       # B H W C
    src = sample_row(source, centers + window_offsets(x.device, r)).permute(0, 3, 4, 2, 1)    # B H W K C
    src = layer.norm1(src)
    k, v = layer.k_proj(src), layer.v_proj(src)
    wq, wk, wv = att.in_proj_weight.chunk(3)
    bq, bk, bv = att.in_proj_bias.chunk(3)
    Q = F.linear(q, wq, bq).view(B, H, W, nh, hd)
    Kt = F.linear(k, wk, bk).view(B, H, W, -1, nh, hd)
    V = F.linear(v, wv, bv).view(B, H, W, -1, nh, hd)
    a = torch.softmax(torch.einsum('bhwnd,bhwknd->bhwnk', Q, Kt) / math.sqrt(hd), -1)
    m = torch.einsum('bhwnk,bhwknd->bhwnd', a, V).reshape(B, H, W, C)
    m = layer.merge(att.out_proj(m)).permute(0, 3, 1, 2)
    m = layer.mlp(torch.cat([x, m], 1).permute(0, 2, 3, 1))
    return x + layer.norm2(m).permute(0, 3, 1, 2)


class TemporalAttention(nn.Module):
    """temporal attention: every cell of the current features attends to the k x k neighbourhood of the previous
    frame's features (same view), then an MLP; both residual with zero-initialised outputs (starts as identity)"""

    def __init__(self, dim=96, heads=4, k=5):
        super().__init__()
        self.k, self.h = k, heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        for lin in (self.out, self.mlp[-1]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, prev):
        B, C, H, W = x.shape
        d, K = C // self.h, self.k * self.k
        xq = x.permute(0, 2, 3, 1)
        q = self.q(self.norm_q(xq)).permute(0, 3, 1, 2).reshape(B, self.h, d, 1, H, W)
        kv = self.kv(self.norm_kv(prev.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        kv = F.unfold(kv, self.k, padding=self.k // 2).view(B, 2, self.h, d, K, H, W)
        valid = F.unfold(torch.ones_like(x[:, :1]), self.k, padding=self.k // 2).view(B, 1, 1, K, H, W) > 0
        logit = (q * kv[:, 0]).sum(2, keepdim=True) / math.sqrt(d)
        a = torch.softmax(logit.masked_fill(~valid, float('-inf')), dim=3)
        o = (a * kv[:, 1]).sum(3).reshape(B, C, H, W).permute(0, 2, 3, 1)
        y = xq + self.out(o)
        y = y + self.mlp(y)
        return y.permute(0, 3, 1, 2)


class TemporalGPS(nn.Module):
    def __init__(self, teacher, iters=8, radius=R, update_dilation=1, full_attention=False, temporal_attention=None):
        """full_attention: the original epipolar attention (whole row, LoFTR layers + residual block) and cnet run on
        the whole map every frame; no proposal MLP, the local cost-volume window is centred on the last match.
        temporal_attention (with full_attention): 'mix' = epipolar attention, then temporal attention to the previous
        frame's features; 'only' = temporal attention in place of the epipolar attention layers"""
        super().__init__()
        self.r = radius
        self.full_attention = full_attention
        self.temporal_attention = temporal_attention
        self.update_dilation = update_dilation  # cells added around the STSS changes before updating (0: STSS cells only)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.iters = iters
        rs = teacher.raft_stereo
        self.n_down = rs.args.n_downsample
        self.stss = STSS()
        self.proposal = ProposalMLP()
        self.att_layers = copy.deepcopy(teacher.loftr_coarse.layers)
        self.att_resid = copy.deepcopy(teacher.loftr_coarse.resid)
        if full_attention:
            self.att_full = copy.deepcopy(teacher.loftr_coarse)
        if temporal_attention:
            self.temporal_att = TemporalAttention(teacher.loftr_coarse.resid.conv1.in_channels if hasattr(teacher.loftr_coarse.resid, 'conv1') else 96)
        self.cnet = copy.deepcopy(rs.cnet)
        self.zqr = copy.deepcopy(rs.context_zqr_convs)
        self.update_block = copy.deepcopy(rs.update_module.update_block)
        # the GRU now reads 2r+1 local cost-volume values instead of 4 pyramid levels x 9: the finest pyramid
        # level (offsets -4..4) initialises the matching centre channels, the rest start at zero
        old = self.update_block.encoder.convc1
        new = nn.Conv2d(2 * radius + 1, old.out_channels, 1)
        with torch.no_grad():
            new.weight.zero_()
            k = min(radius, 4)
            new.weight[:, radius - k:radius + k + 1] = old.weight[:, 4 - k:4 + k + 1]
            new.bias.copy_(old.bias)
        self.update_block.encoder.convc1 = new
        self.regressor = copy.deepcopy(teacher.gs_parm_regresser)
        # copies of the frozen teacher modules are trained
        for name, p in self.named_parameters():
            if not name.startswith('teacher.'):
                p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    # ---------------- helpers ----------------
    @staticmethod
    def inputs(data):
        L, Rv = data['lmain'], data['rmain']
        cat = lambda k: torch.cat([L[k], Rv[k]], 0)
        return {'image': cat('img'), 'mask': cat('mask'), 'intr': cat('intr'), 'ref_intr': cat('ref_intr'),
                'extr': cat('extr'), 'tf_x': torch.cat([L['Tf_x'].view(1), Rv['Tf_x'].view(1)]),
                'flow_init': cat('flow_init') if 'flow_init' in L else None}

    def upsample(self, flow, mask):
        N, D, H, W = flow.shape
        f = 2 ** self.n_down
        mask = torch.softmax(mask.view(N, 1, 9, f, f, H, W), dim=2)
        up = F.unfold(f * flow, [3, 3], padding=1).view(N, D, 9, 1, 1, H, W)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(N, D, f * H, f * W)

    @staticmethod
    def to_depth(flow, inp):
        offset = (inp['ref_intr'][:, 0, 2] - inp['intr'][:, 0, 2])[:, None, None, None]
        return -(offset - flow) / inp['tf_x'][:, None, None, None]

    def gaussians(self, regressor, inp, depth, img_feat):
        rot, scale, opacity, res = regressor(inp['image'], depth, img_feat)
        depth = depth + res
        xyz = depth2pc(depth, inp['extr'], inp['intr'])
        return {'xyz': xyz, 'rot': rot, 'scale': scale, 'opacity': opacity, 'img': inp['image'],
                'valid': (inp['mask'][:, :1] > 0.5).flatten(1), 'depth': depth}

    # ---------------- keyframe: the per-frame model ----------------
    @torch.no_grad()
    def keyframe(self, data):
        t, inp = self.teacher, self.inputs(data)
        feat = t.img_encoder(inp['image'])
        c0, c1 = t.loftr_coarse(*feat[2].split(1), None, None)
        feat_cs = torch.cat([c0, c1], 0)
        rs = t.raft_stereo
        *cl, fm1, fm2 = rs.cnet(feat_cs)
        net = torch.tanh(cl[0][0])
        ctx = list(rs.context_zqr_convs[0](torch.relu(cl[0][1])).split(rs.context_zqr_convs[0].out_channels // 3, 1))
        fmap, fmap_o = torch.cat([fm1, fm2], 0).float(), torch.cat([fm2, fm1], 0).float()
        corr_fn = CorrBlock1D(fmap, fmap_o, radius=rs.args.corr_radius, num_levels=rs.args.corr_levels)
        coords0 = coords_grid(2, *net.shape[-2:]).to(net.device)
        fi = downflow8(inp['flow_init'])
        coords1 = coords0 + torch.cat([fi, torch.zeros_like(fi)], 1)
        for _ in range(self.iters):
            corr = corr_fn(coords1)
            nl, up_mask, delta = rs.update_module.update_block([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
            net = nl[0]
            delta[:, 1] = 0
            coords1 = coords1 + delta
        flow = self.upsample(coords1 - coords0, up_mask)[:, :1]
        g = self.gaussians(t.gs_parm_regresser, inp, self.to_depth(flow, inp), (feat[0], feat[1], feat_cs))
        centers = coords1[:, :1]
        state = {'f3': feat[2], 'feat_cs': feat_cs, 'centers': centers, 'cv': local_cost_volume(fmap, fmap_o, centers, self.r),
                 'net': net, 'coords1': coords1, 'flow': flow, 'g': g, 'image': inp['image']}
        return state

    # ---------------- frame t ----------------
    def step(self, data, state, mask=None, use_proposal=True):
        """mask: None -> STSS prediction, else a given (B,1,h,w) change mask (before dilation)"""
        inp = self.inputs(data)
        with torch.no_grad():
            feat = self.teacher.img_encoder(inp['image'])  # shared, frozen encoder
        f3 = feat[2]
        logits = self.stss(f3, state['f3'])
        changed = (torch.sigmoid(logits) > 0.5).float() if mask is None else mask
        k = self.update_dilation
        M = F.max_pool2d(changed, 2 * k + 1, 1, k) if k > 0 else changed  # cells that get new values
        Mb = M > 0.5

        h, w = f3.shape[-2:]
        c_prev = state['centers']
        if self.full_attention:
            # original epipolar attention on the whole map (current frame, both views); window centre = last match
            p_star = state['coords1'][:, :1]
            feat_cs = self.attend_full(f3, state['feat_cs'])
        else:
            # correspondence proposal on the epipolar line
            xg = torch.arange(w, device=f3.device, dtype=torch.float32).view(1, 1, 1, w)
            f_o = sample_row(swap_views(f3), c_prev).squeeze(2)
            offset = self.proposal(f3, state['f3'], f_o, c_prev - xg)
            p_star = c_prev + offset if use_proposal else c_prev

            # local epipolar attention around p* (left queries look into the right view and vice versa)
            f0, f1 = f3[0:1], f3[1:2]
            for layer in self.att_layers:
                f0 = local_cross_attention(layer, f0, f1, p_star[0:1], self.r)
                f1 = local_cross_attention(layer, f1, f0, p_star[1:2], self.r)
            att = self.att_resid(torch.cat([f0, f1], 0))  # per-sample norms: same as one call per view
            feat_cs = torch.where(Mb, att, state['feat_cs'])

        # local cost volume, previous one where unchanged
        *cl, fm1, fm2 = self.cnet(feat_cs)
        ctx = list(self.zqr[0](torch.relu(cl[0][1])).split(self.zqr[0].out_channels // 3, 1))
        fmap, fmap_o = torch.cat([fm1, fm2], 0).float(), torch.cat([fm2, fm1], 0).float()
        centers = torch.where(Mb, p_star, c_prev)
        cv = torch.where(Mb, local_cost_volume(fmap, fmap_o, p_star, self.r), state['cv'])

        # GRU, updates only the changed region, from the previous state
        net, coords1 = state['net'], state['coords1']
        coords0 = coords_grid(2, h, w).to(f3.device)
        for _ in range(self.iters):
            corr = lookup(cv, centers, coords1[:, :1], self.r)
            nl, up_mask, delta = self.update_block([net], [ctx], corr, coords1 - coords0, iter32=False, iter16=False)
            net = torch.where(Mb, nl[0], net)
            delta = torch.cat([delta[:, :1], torch.zeros_like(delta[:, 1:])], 1)
            coords1 = coords1 + delta * M
        Mf = F.interpolate(M, scale_factor=2 ** self.n_down, mode='nearest') > 0.5
        flow = torch.where(Mf, self.upsample(coords1 - coords0, up_mask)[:, :1], state['flow'])

        # Gaussians: new ones in the changed region, previous ones elsewhere
        g_new = self.gaussians(self.regressor, inp, self.to_depth(flow, inp), (feat[0], feat[1], feat_cs))
        gp = state['g']
        Mp = Mf.flatten(1)
        g = {'xyz': torch.where(Mp[..., None], g_new['xyz'], gp['xyz']),
             'rot': torch.where(Mf, g_new['rot'], gp['rot']),
             'scale': torch.where(Mf, g_new['scale'], gp['scale']),
             'opacity': torch.where(Mf, g_new['opacity'], gp['opacity']),
             'img': torch.where(Mf, g_new['img'], gp['img']),
             'valid': g_new['valid'], 'depth': torch.where(Mf, g_new['depth'], gp['depth'])}

        # the STSS reference of every cell (features and image) is the frame its kept values come from: it only moves
        # to frame t where the cell was updated, so a slow change keeps growing against it until the cell is updated
        new_state = {'f3': torch.where(Mb, f3, state['f3']), 'feat_cs': feat_cs, 'centers': centers, 'cv': cv, 'net': net,
                     'coords1': coords1, 'flow': flow, 'g': g, 'image': torch.where(Mf, inp['image'], state['image'])}
        out = {'g': g, 'flow': flow, 'p_star': p_star, 'logits': logits, 'mask': M, 'changed': changed,
               'fmap': fmap, 'ctx': ctx}
        return out, new_state


    def attend_full(self, f3, prev_feat_cs):
        """full_attention models: epipolar attention (whole rows) and / or temporal attention on the whole map"""
        if self.temporal_attention == 'only':
            return self.att_full.resid(self.temporal_att(f3, prev_feat_cs))
        c0, c1 = self.att_full(f3[0:1], f3[1:2], None, None)
        feat_cs = torch.cat([c0, c1], 0)
        return self.temporal_att(feat_cs, prev_feat_cs) if self.temporal_attention == 'mix' else feat_cs

    # ---------------- patch mode (lib/temporal_patch.py) ----------------
    @torch.no_grad()
    def prepare_patch(self, data, state):
        """keyframe state for patch mode: records the full-map normalisation statistics (FrozenNorm, see
        lib/temporal_sparse.freeze_norms) with one dense pass and keeps the cnet outputs as caches"""
        from lib.temporal_sparse import set_norm_mode
        set_norm_mode(self, 'record')
        out, _ = self.step(data, state, mask=torch.zeros_like(state['centers']))
        set_norm_mode(self, 'frozen')
        st = dict(state)
        st['fmap'], st['ctx'] = out['fmap'], [c.clone() for c in out['ctx']]
        st['coords0'] = coords_grid(2, *state['centers'].shape[-2:]).to(state['centers'].device)
        return st

    def step_patch(self, data, state, k, halo, mask=None, use_proposal=True):
        """step with the conv parts on k x k-cell patches (+ halo); point-wise parts dense"""
        from lib.temporal_patch import patch_tiles, patch_convs, patch_groups
        inp = self.inputs(data)
        with torch.no_grad():
            feat = self.teacher.img_encoder(inp['image'])
        f3 = feat[2]
        logits = self.stss(f3, state['f3'])
        changed = (torch.sigmoid(logits) > 0.5).float() if mask is None else mask
        M = changed > 0.5
        h, w = f3.shape[-2:]
        xg = torch.arange(w, device=f3.device, dtype=torch.float32).view(1, 1, 1, w)
        c_prev = state['centers']
        f_o = sample_row(swap_views(f3), c_prev).squeeze(2)
        offset = self.proposal(f3, state['f3'], f_o, c_prev - xg)
        p_star = c_prev + offset if use_proposal else c_prev
        f0, f1 = f3[0:1], f3[1:2]
        for layer in self.att_layers:
            f0 = local_cross_attention(layer, f0, f1, p_star[0:1], self.r)
            f1 = local_cross_attention(layer, f1, f0, p_star[1:2], self.r)
        idx = patch_tiles(M, k)
        out = {'p_star': p_star, 'logits': logits, 'mask': M.float(), 'changed': changed}
        if idx.shape[0] == 0:
            return dict(out, g=state['g'], flow=state['flow'], groups=torch.zeros_like(state['flow'], dtype=torch.long)), dict(state)
        res = patch_convs(self, M, idx, k, halo, torch.cat([f0, f1], 0), p_star, feat, inp, state)
        Mf = res.pop('Mf')
        new_state = dict(state)
        new_state.update(res)
        new_state['f3'] = torch.where(M, f3, state['f3'])
        new_state['image'] = torch.where(Mf, inp['image'], state['image'])
        return dict(out, g=res['g'], flow=res['flow'], groups=patch_groups(M, idx, k)), new_state


def border_depth_loss(depth, updated, image, valid, gamma=10.0):
    """depth mismatch on the border between updated and kept pixels: mean |log depth| difference of 4-neighbour pairs
    with one pixel on each side, weighted by exp(-gamma * colour difference) so that real depth edges, which come with a
    colour edge, are not pulled together. Kept depths come from the (detached) state, so only the new side moves.
    depth, updated, valid: (B,1,H,W); image: (B,3,H,W) in [-1, 1]"""
    z = depth.clamp(min=1e-3).log()
    num, den = 0, 0
    for dim in (2, 3):
        n = z.shape[dim] - 1
        a, b = (lambda x: x.narrow(dim, 0, n)), (lambda x: x.narrow(dim, 1, n))
        cross = (a(updated) != b(updated)) & a(valid) & b(valid)  # updated: bool, or group ids (patch mode)
        w = torch.exp(-gamma * (a(image) - b(image)).abs().mean(1, keepdim=True) / 2)
        num = num + (w * (a(z) - b(z)).abs() * cross).sum()
        den = den + cross.sum()
    return num / den.clamp(min=1)


def detach_state(state):
    def det(v):
        if isinstance(v, dict):
            return {kk: vv.detach() for kk, vv in v.items()}
        return [vv.detach() for vv in v] if isinstance(v, list) else v.detach()
    return {k: det(v) for k, v in state.items()}


def fill_render_data(data, g):
    """put Gaussians (batch = [left, right]) into the data dict layout of lib.GaussianRender.pts2render"""
    for i, view in enumerate(['lmain', 'rmain']):
        data[view]['xyz'] = g['xyz'][i:i + 1]
        data[view]['img'] = g['img'][i:i + 1]
        data[view]['rot_maps'] = g['rot'][i:i + 1]
        data[view]['scale_maps'] = g['scale'][i:i + 1]
        data[view]['opacity_maps'] = g['opacity'][i:i + 1]
        data[view]['pts_valid'] = g['valid'][i:i + 1]
    return data
