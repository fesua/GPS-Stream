import torch
import torch.nn as nn

# GroupNorm with statistics recorded on one input (the keyframe) and reused afterwards. In 'frozen' mode the layer is a
# per-sample, per-channel affine map, which is what tiled / delta (DeltaCNN) execution needs: the statistics of a tile
# or of a sparse update must not depend on what the layer sees in that call. Same idea as FrozenNorm in the GPS_plus
# method code (lib/temporal_sparse.py), kept minimal here.


class FrozenGroupNorm(nn.Module):
    def __init__(self, gn):
        super().__init__()
        self.G, self.C, self.eps = gn.num_groups, gn.num_channels, gn.eps
        self.weight, self.bias = gn.weight, gn.bias  # shared with the original layer (no copy)
        self.mode = 'live'  # 'live' = GroupNorm, 'record' = GroupNorm + store the statistics, 'frozen' = use the stored ones
        self.mean = self.var = None  # (N, G, 1) of the recorded batch
        self.sample_ids = None  # frozen mode with a different batch (tiles): index of the recorded sample for each input

    def forward(self, x):
        N, C, H, W = x.shape
        xg = x.reshape(N, self.G, -1)
        if self.mode == 'frozen':
            ids = self.sample_ids if self.sample_ids is not None else torch.arange(N, device=x.device)
            mean, var = self.mean[ids], self.var[ids]
        else:
            mean, var = xg.mean(-1, keepdim=True), xg.var(-1, unbiased=False, keepdim=True)
            if self.mode == 'record':
                self.mean, self.var = mean.detach(), var.detach()
        y = ((xg - mean) * torch.rsqrt(var + self.eps)).reshape(N, C, H, W)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def channel_affine(self, sample):
        """frozen statistics of one recorded sample as per-channel (scale, offset): y = x * scale + offset"""
        rep = self.C // self.G
        mean, var = self.mean[sample, :, 0].repeat_interleave(rep), self.var[sample, :, 0].repeat_interleave(rep)
        scale = self.weight * torch.rsqrt(var + self.eps)
        return scale, self.bias - mean * scale


def freeze_group_norms(module, _seen=None):
    """replace every nn.GroupNorm inside module by a FrozenGroupNorm (parameters shared); a GroupNorm referenced twice
    (ResidualBlock.norm3 is also inside .downsample) gets one wrapper; returns the list of wrappers"""
    seen = {} if _seen is None else _seen
    out = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            if id(child) not in seen:
                seen[id(child)] = FrozenGroupNorm(child); out.append(seen[id(child)])
            setattr(module, name, seen[id(child)])
        else:
            out += freeze_group_norms(child, seen)
    return out


def set_norm_mode(module, mode, sample_ids=None):
    for m in module.modules():
        if isinstance(m, FrozenGroupNorm):
            m.mode = mode
            m.sample_ids = sample_ids
