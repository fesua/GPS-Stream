import torch
import torch.nn.functional as F
from lib.temporal_gps import TemporalGPS
from lib.temporal_loader import change_target

# Step 3 of the patch pipeline: dynamic regions estimated patch-wise with the GPS-Gaussian+ recipe.
#   1. encoder: the teacher's encoder on the whole frame (inside model.keyframe)
#   2. STSS (trained, from TemporalGPS) marks changed cells against each cell's last-update reference features
#   3. patches = P x P pixel tiles (P / 8 cells, aligned to the cell grid) that contain a changed cell; inside them depth
#      and Gaussians come from the per-frame recipe (full epipolar attention, full cost volume, fresh GRU state,
#      inverse-depth 0.5 start, 8 GRU iterations, convex upsampling, regressor); outside, the previous frame's values
#   4. (later) boundary interpolation - not done here: tiles are simply pasted
# Phase A emulates the patch computation densely (the teacher runs on the whole frame, values are taken inside the
# tiles), which equals a patch computation with a full context margin. mask_mode: 'stss' (prediction) or 'target'
# (the image-difference target STSS is trained on: mean colour change of the cell > 8/255 against its reference).


class PatchStep3:
    def __init__(self, model: TemporalGPS, patch=64, mask_mode='stss', dilate=0, footprint='tile', margin=0, key_model=None):
        """patch: tile side in px (8 = the changed cells themselves); dilate: cells added around the changed cells before
        tiling; footprint: 'tile' (P x P grid tiles containing a changed cell), 'band' (full-width row bands of height P
        containing a changed cell: the natural unit of rectified stereo), 'bbox' (bounding box of each connected group of
        changed cells, grown by `margin` cells; patch is ignored)"""
        assert patch % 8 == 0
        self.m, self.c, self.mask_mode, self.dilate, self.footprint, self.margin = model, patch // 8, mask_mode, dilate, footprint, margin
        self.key = key_model or model  # the start frame comes from this (frozen, original) per-frame network; later frames from `model`

    @staticmethod
    def footprint_mask(changed, c, footprint='tile', margin=0):
        """(B,1,h,w) float changed cells -> (B,1,h,w) bool cells to update"""
        if footprint == 'tile':
            if c <= 1:
                return changed > 0.5
            h, w = changed.shape[-2:]
            pooled = F.max_pool2d(changed, c, c, ceil_mode=True)   # tiles anchored at (0, 0); edge tiles are partial when c does not divide the map (192 px on 1024)
            return F.interpolate(pooled, scale_factor=c, mode='nearest')[..., :h, :w] > 0.5
        if footprint == 'band':
            rows = F.max_pool2d(changed, (c, changed.shape[-1]), (c, changed.shape[-1]))        # (B,1,h/c,1)
            return F.interpolate(rows, size=changed.shape[-2:], mode='nearest') > 0.5
        if footprint == 'bbox':
            import numpy as np
            import cv2
            out = torch.zeros_like(changed, dtype=torch.bool)
            for b in range(changed.shape[0]):
                m = (changed[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                h, w = m.shape
                for i in range(1, n):
                    x, y, bw, bh = stats[i, :4]
                    out[b, 0, max(y - margin, 0):min(y + bh + margin, h), max(x - margin, 0):min(x + bw + margin, w)] = True
            return out
        raise ValueError(footprint)

    @torch.no_grad()
    def keyframe(self, data):
        st = self.key.keyframe(data)
        return {'f3': st['f3'], 'image': st['image'], 'flow': st['flow'], 'g': st['g']}

    @torch.no_grad()
    def step(self, data, state):
        m = self.m
        teach = m.keyframe(data)                       # the per-frame recipe on the whole frame (encoder included)
        inp = m.inputs(data)
        f3 = teach['f3']
        if self.mask_mode == 'target':
            changed = change_target(inp['image'], state['image'])
        else:
            changed = (torch.sigmoid(m.stss(f3, state['f3'])) > 0.5).float()
        c, k = self.c, self.dilate
        upd = F.max_pool2d(changed, 2 * k + 1, 1, k) if k > 0 else changed
        M = self.footprint_mask(upd, c, self.footprint, self.margin)
        Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
        flow = torch.where(Mf, teach['flow'], state['flow'])
        gn, gp = teach['g'], state['g']
        Mp = Mf.flatten(1)
        g = {'xyz': torch.where(Mp[..., None], gn['xyz'], gp['xyz']), 'valid': gn['valid']}
        for k in ('rot', 'scale', 'opacity', 'img', 'depth'):
            g[k] = torch.where(Mf, gn[k], gp[k])
        new_state = {'f3': torch.where(M, f3, state['f3']), 'image': torch.where(Mf, inp['image'], state['image']), 'flow': flow, 'g': g}
        out = {'g': g, 'flow': flow, 'mask': M.float(), 'changed': changed, 'teach_flow': teach['flow'], 'teach_g': teach['g']}
        return out, new_state
