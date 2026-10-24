import torch
import torch.nn as nn
import torch.nn.functional as F
from deltacnn.sparse_layers import (DCConv2d, DCBatchNorm2d, DCAdd, DCActivation, DCConcatenate, DCSparsify, DCDensify,
                                    DCUpsamplingNearest2d, DCModule, DCThreshold)

from lib.frozen_norm import FrozenGroupNorm

# DeltaCNN (Parger et al., CVPR 2022) port of the GPS-Gaussian+ CNNs: UnetExtractor (image / depth encoder) and
# GSRegresser. Every conv / norm / activation / add / concat / upsampling is a DeltaCNN layer: the first frame is
# processed densely, later frames propagate only the pixels whose input changed (deltas), layer by layer.
# - GroupNorm -> per-channel affine from statistics frozen at the keyframe (lib.frozen_norm), folded into DCBatchNorm2d.
#   A BatchNorm holds one statistic per channel, so one model instance per view (sample) keeps the per-view statistics
#   of the dense model; averaging the two views' statistics into one instance costs ~0.1 dB on one of the views
# - bilinear x2 (align_corners=False) -> nearest x2 + fixed depthwise 3x3 filter [1,2,1]/4 (x) [1,2,1]/4 + a per-pixel
#   scale of the outermost row / column (the zero-padded filter sums 3/4 of the weights there; bilinear clamps) = exact
# - head activations (normalise / softplus+clamp / sigmoid / tanh) are applied densely on the 9 output channels
# Inputs must be the "patchwork" inputs of the reuse scheme (current values in updated cells, reference values
# elsewhere) so that the deltas are exactly zero where nothing is updated; delta threshold 0 then gives the dense result.


def fold_norm(dc_bn, fgn, sample=None):
    """DCBatchNorm2d <- FrozenGroupNorm statistics of one recorded sample (sample = index in the recorded batch), or
    averaged over the recorded samples (sample None). A BatchNorm holds one statistic per channel, so a model instance
    serves one sample; averaging the two views' statistics costs ~0.1 dB on the view whose statistics differ more."""
    C, G = fgn.C, fgn.G
    mean, var = (fgn.mean.mean(0), fgn.var.mean(0)) if sample is None else (fgn.mean[sample], fgn.var[sample])
    mean, var = mean[:, 0].repeat_interleave(C // G), var[:, 0].repeat_interleave(C // G)
    with torch.no_grad():
        dc_bn.running_mean.copy_(mean); dc_bn.running_var.copy_(var)
        dc_bn.weight.copy_(fgn.weight); dc_bn.bias.copy_(fgn.bias)
    dc_bn.eps = fgn.eps
    dc_bn.source, dc_bn.sample = fgn, sample  # kept for re-folding after the statistics are re-recorded (new sequence)
    return dc_bn


def dc_conv(conv):
    dc = DCConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        dc.weight.copy_(conv.weight)
        if conv.bias is not None:
            dc.bias.copy_(conv.bias)
    return dc


class DCBilinearUp(DCModule):
    """nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) = nearest x2 followed by the depthwise 3x3
    filter [1,2,1]/4 (x) [1,2,1]/4. The filter is zero-padded, so the outermost output row / column sums only 3/4 of the
    weights (corners 9/16) where bilinear clamps the border; a fixed per-pixel scale (4/3 on the edges, 16/9 in the
    corners) restores the clamped value exactly. The scale is linear, so it applies to the deltas as it does to the
    dense values, and the update mask is unchanged."""

    def __init__(self, C):
        super().__init__()
        self.up = DCUpsamplingNearest2d(scale_factor=2)
        self.f = DCConv2d(C, C, 3, 1, 1, groups=C, bias=False)
        w = torch.tensor([0.25, 0.5, 0.25])
        with torch.no_grad():
            self.f.weight.copy_((w[:, None] * w[None, :]).expand(C, 1, 3, 3))
        self.border = {}

    def border_scale(self, val):
        H, W = val.shape[-2:]
        s = self.border.get((H, W))
        if s is None or s.device != val.device:
            s = torch.ones(1, 1, H, W, device=val.device, dtype=val.dtype)
            s[..., [0, -1], :] *= 4 / 3
            s[..., :, [0, -1]] *= 4 / 3
            self.border[(H, W)] = s
        return s

    def forward(self, x):
        y = self.f(self.up(x))
        if isinstance(y, tuple):
            val, mask = y
            return (val * self.border_scale(val)).contiguous(memory_format=torch.channels_last), mask
        return (y * self.border_scale(y)).contiguous(memory_format=torch.channels_last)


def dc_bilinear_up(C):
    return DCBilinearUp(C)


class DCResidualBlock(DCModule):
    def __init__(self, rb, sample=None):
        super().__init__()
        self.c1, self.n1, self.a1 = dc_conv(rb.conv1), fold_norm(DCBatchNorm2d(rb.conv1.out_channels), rb.norm1, sample), DCActivation(activation='relu')
        self.c2, self.n2, self.a2 = dc_conv(rb.conv2), fold_norm(DCBatchNorm2d(rb.conv2.out_channels), rb.norm2, sample), DCActivation(activation='relu')
        self.down = None
        if rb.downsample is not None:
            self.down = nn.Sequential(dc_conv(rb.downsample[0]), fold_norm(DCBatchNorm2d(rb.conv1.out_channels), rb.downsample[1], sample))
        self.add = DCAdd(activation='relu')

    def forward(self, x):
        y = self.a2(self.n2(self.c2(self.a1(self.n1(self.c1(x))))))
        return self.add(self.down(x) if self.down is not None else x, y)


class DCUnet(DCModule):
    """UnetExtractor; forward(x) with x dense (B, C, H, W) -> sparse f1, f2, f3"""

    def __init__(self, unet, threshold=0.0, sample=None):
        """sample: index of the recorded sample whose normalisation statistics this instance holds (None: averaged)"""
        super().__init__()
        self.sp = DCSparsify(delta_threshold=threshold)
        self.in_c, self.in_n, self.in_a = dc_conv(unet.in_ds[0]), fold_norm(DCBatchNorm2d(32), unet.in_ds[1], sample), DCActivation(activation='relu')
        self.res1 = nn.Sequential(*[DCResidualBlock(b, sample) for b in unet.res1])
        self.res2 = nn.Sequential(*[DCResidualBlock(b, sample) for b in unet.res2])
        self.res3 = nn.Sequential(*[DCResidualBlock(b, sample) for b in unet.res3])

    def forward(self, x):
        x = self.in_a(self.in_n(self.in_c(self.sp(x))))
        f1 = self.res1(x); f2 = self.res2(f1); f3 = self.res3(f2)
        return f1, f2, f3


class DCRegressor(DCModule):
    """GSRegresser; forward(img, depth, (f1, f2, f3)) -> raw heads (B, 9, H, W) dense. As in the model, f3 must be the
    1/8 features AFTER the epipolar attention (feat_cs), not the encoder's raw 1/8 output."""

    def __init__(self, reg, threshold=0.0, dense_feat=False, sample=None, feat_mode=None):
        """feat_mode: how (f1, f2, f3) arrive - 'dense' (dense patchwork tensors, sparsified here), 'mixed' (f1, f2 sparse
        from the image DCUnet, f3 = the attended features as a dense patchwork tensor, sparsified here) or 'sparse' (all
        three sparse); dense_feat=True is the old name of 'dense'. sample: index of the recorded sample whose
        normalisation statistics this instance holds (None: averaged)"""
        super().__init__()
        self.feat_mode = feat_mode or ('dense' if dense_feat else 'sparse')
        n_sp = {'sparse': 0, 'mixed': 1, 'dense': 3}[self.feat_mode]
        self.sp_img, self.sp_depth = DCSparsify(delta_threshold=threshold), DCSparsify(delta_threshold=threshold)
        self.sp_feat = nn.ModuleList([DCSparsify(delta_threshold=threshold) for _ in range(n_sp)]) if n_sp else None
        self.denc = DCUnet(reg.depth_encoder, threshold, sample)
        self.cat3, self.cat2, self.cat2b, self.cat1, self.cat1b, self.catt, self.catt2 = [DCConcatenate() for _ in range(7)]
        self.dec3 = nn.Sequential(*[DCResidualBlock(b, sample) for b in reg.decoder3]); self.up3 = dc_bilinear_up(reg.decoder3[-1].conv2.out_channels)
        self.dec2 = nn.Sequential(*[DCResidualBlock(b, sample) for b in reg.decoder2]); self.up2 = dc_bilinear_up(reg.decoder2[-1].conv2.out_channels)
        self.dec1 = nn.Sequential(*[DCResidualBlock(b, sample) for b in reg.decoder1]); self.up1 = dc_bilinear_up(reg.decoder1[-1].conv2.out_channels)
        self.out_c, self.out_a = dc_conv(reg.out_conv), DCActivation(activation='relu')
        self.heads = nn.ModuleList()
        for h in (reg.rot_head, reg.scale_head, reg.opacity_head, reg.depth_head):
            self.heads.append(nn.Sequential(dc_conv(h[0]), DCActivation(activation='relu'), dc_conv(h[2]), DCDensify()))

    def forward(self, img, depth, img_feat):
        if self.feat_mode == 'dense':
            img_feat = [sp(x) for sp, x in zip(self.sp_feat, img_feat)]
        elif self.feat_mode == 'mixed':
            img_feat = [img_feat[0], img_feat[1], self.sp_feat[0](img_feat[2])]
        d1, d2, d3 = self.denc(depth)
        up3 = self.up3(self.dec3(self.cat3(img_feat[2], d3)))
        up2 = self.up2(self.dec2(self.cat2(up3, self.cat2b(img_feat[1], d2))))
        up1 = self.up1(self.dec1(self.cat1(up2, self.cat1b(img_feat[0], d1))))
        o = self.out_a(self.out_c(self.catt(self.catt2(up1, self.sp_img(img)), self.sp_depth(depth))))
        return torch.cat([h(o) for h in self.heads], 1)


class NearestFilterUp(nn.Module):
    """PyTorch version of the uncorrected DCBilinearUp (nearest x2 + depthwise [1,2,1]/4 filter, zero padding, no border
    scale) for the dense model: measures what the zero-padded border costs (`test_baseline.py --up nearest`)"""

    def forward(self, x):
        C = x.shape[1]
        w = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype)
        return F.conv2d(F.interpolate(x, scale_factor=2, mode='nearest'), (w[:, None] * w[None, :]).expand(C, 1, 3, 3), None, 1, 1, 1, C)


def prepare(model):
    """DeltaCNN model: channels_last, filters processed, eval"""
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    model.process_filters()
    return model


def cl(x):
    return x.contiguous(memory_format=torch.channels_last)
