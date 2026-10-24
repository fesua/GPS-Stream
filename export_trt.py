import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.GaussianRender import pts2render

# python export_trt.py --ckpt experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth --iters 8
# Exports everything before rasterization (stereo depth + Gaussian parameter maps) to ONNX, builds fp32 / fp16
# TensorRT engines next to the checkpoint and benchmarks them against PyTorch on one test sample.
# Requires: pip install tensorrt-cu11 onnx -c <numpy pinned to the installed version>

LOG = trt.Logger(trt.Logger.WARNING)
OUTPUTS = ['xyz', 'rot', 'scale', 'opacity']


def depth2pc_export(depth, extrinsic, intrinsic):
    # lib.utils.depth2pc rewritten for ONNX/TensorRT: no in-place indexed writes, no mixed None/int indexing
    B, _, H, W = depth.shape
    y = torch.arange(H, dtype=torch.float32, device=depth.device).view(1, H, 1) + 0.5
    x = torch.arange(W, dtype=torch.float32, device=depth.device).view(1, 1, W) + 0.5
    fx, fy = intrinsic[:, 0, 0].view(B, 1, 1), intrinsic[:, 1, 1].view(B, 1, 1)
    cx, cy = intrinsic[:, 0, 2].view(B, 1, 1), intrinsic[:, 1, 2].view(B, 1, 1)
    z = 1.0 / (depth[:, 0] + 1e-8)
    pts = torch.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], dim=1).view(B, 3, H * W)
    rot, trans = extrinsic[:, :, :3], extrinsic[:, :, 3:]
    return torch.matmul(rot.transpose(1, 2), pts - trans).transpose(1, 2)


class ExplicitGroupNorm(nn.Module):
    # nn.GroupNorm as plain reductions; the opset-17 export (InstanceNormalization on a reshaped tensor) gives
    # wrong results in TensorRT inside the full graph
    def __init__(self, gn):
        super().__init__()
        self.g, self.eps, self.weight, self.bias = gn.num_groups, gn.eps, gn.weight, gn.bias

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, self.g, -1)
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = ((x - mean) / torch.sqrt(var + self.eps)).view(N, C, H, W)
        return x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)


def explicit_group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, ExplicitGroupNorm(child))
        else:
            explicit_group_norm(child)
    return module


class SourceNet(nn.Module):
    """source pair -> Gaussian parameter maps (RtStereoHumanModel.forward without rendering), batch = [lmain, rmain]"""

    def __init__(self, model, iters):
        super().__init__()
        self.m, self.iters = model, iters

    def forward(self, image, flow_init, intr, ref_intr, extr, tf_x):
        m = self.m
        img_feat = m.img_encoder(image)
        c0, c1 = m.loftr_coarse(*img_feat[2].split(1), None, None)
        feat_cs = torch.cat((c0, c1), 0)
        img_feat = img_feat[0], img_feat[1], feat_cs
        flow = torch.cat(m.raft_stereo(feat_cs, flow_init=flow_init, iters=self.iters, test_mode=True), 0)
        # lib.utils.flow2depth for both views at once
        offset = (ref_intr[:, 0, 2] - intr[:, 0, 2])[:, None, None, None]
        depth = -(offset - flow) / tf_x[:, None, None, None]
        rot, scale, opacity, res_depth = m.gs_parm_regresser(image, depth, img_feat)
        return depth2pc_export(depth + res_depth, extr, intr), rot, scale, opacity


def source_inputs(data):
    L, R = data['lmain'], data['rmain']
    inputs = {'image': torch.cat([L['img'], R['img']]), 'flow_init': torch.cat([L['flow_init'], R['flow_init']]),
              'intr': torch.cat([L['intr'], R['intr']]), 'ref_intr': torch.cat([L['ref_intr'], R['ref_intr']]),
              'extr': torch.cat([L['extr'], R['extr']]), 'tf_x': torch.cat([L['Tf_x'], R['Tf_x']]).reshape(2)}
    return {k: v.float().contiguous() for k, v in inputs.items()}


def fill_gaussians(data, xyz, rot, scale, opacity):
    for i, view in enumerate(['lmain', 'rmain']):
        data[view]['xyz'], data[view]['rot_maps'] = xyz[i:i + 1], rot[i:i + 1]
        data[view]['scale_maps'], data[view]['opacity_maps'] = scale[i:i + 1], opacity[i:i + 1]
        data[view]['pts_valid'] = (data[view]['mask'][:, :1] > 0.5).view(1, -1)
    return data


class TrtSourceNet:
    """runs a SourceNet engine on torch CUDA tensors"""

    def __init__(self, engine_path):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(open(engine_path, 'rb').read())
        self.ctx = self.engine.create_execution_context()
        self.outs = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), device='cuda') for n in OUTPUTS}
        for n, t in self.outs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())

    def __call__(self, inputs):
        for n, t in inputs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return [self.outs[n] for n in OUTPUTS]


def build_engine(onnx_path, engine_path, fp16):
    builder = trt.Builder(LOG)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOG)
    assert parser.parse_from_file(onnx_path), [parser.get_error(i) for i in range(parser.num_errors)]
    bc = builder.create_builder_config()
    bc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if fp16:
        bc.set_flag(trt.BuilderFlag.FP16)
    open(engine_path, 'wb').write(builder.build_serialized_network(network, bc))


def timeit(fn, n=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--iters', type=int, default=8, help='RAFT iterations, as in training')
    parser.add_argument('--inv_depth', type=float, default=0.5)
    parser.add_argument('--sample', type=str, default='sear_steak_s2_0250', help='test sample for the benchmark')
    arg = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(arg.ckpt)), 'trt')
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, 'source_net_it%d.onnx' % arg.iters)

    cfg = config()
    cfg.load('config/stage.yaml')
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.raft.val_iters = arg.iters
    cfg.dataset.inverse_depth_init = arg.inv_depth
    model = RtStereoHumanModel(cfg, with_gs_render=True).cuda()
    model.load_state_dict(torch.load(arg.ckpt, map_location='cuda')['network'])
    model.eval()
    model.raft_stereo.freeze_bn()

    seq = arg.sample.rsplit('_', 2)[0] + '_process'
    dataset = StereoHumanDataset(cfg.dataset, phase=seq)
    data = torch.utils.data.default_collate([dataset.get_item(dataset.sample_list.index(arg.sample), novel_id=[3])])
    for view in ['lmain', 'rmain']:
        data[view] = {k: v.cuda() for k, v in data[view].items()}
    for k in ['world_view_transform', 'full_proj_transform', 'camera_center']:
        data['novel_view'][k] = data['novel_view'][k].cuda()
    inputs = source_inputs(data)
    gt = data['novel_view']['img'][0].cuda()

    def render(maps):
        with torch.no_grad():
            return pts2render(fill_gaussians(data, *maps), bg_color=[0, 0, 0])['novel_view']['img_pred'][0].clamp(0, 1)
    psnr = lambda a, b: (10 * torch.log10(1 / ((a - b) ** 2).mean())).item()

    net = SourceNet(model, arg.iters).eval()
    with torch.no_grad():
        ref_maps = net(*inputs.values())
        t_net = timeit(lambda: net(*inputs.values()))
    ref = render(ref_maps)
    t_render = timeit(lambda: render(ref_maps))
    print('PyTorch fp32 : network %.1f ms + render %.1f ms = %.1f ms (%.1f FPS), PSNR vs GT %.2f'
          % (t_net, t_render, t_net + t_render, 1000 / (t_net + t_render), psnr(ref, gt)), flush=True)

    with torch.no_grad():
        torch.onnx.export(SourceNet(explicit_group_norm(model), arg.iters).eval(), tuple(inputs.values()), onnx_path,
                          opset_version=17, input_names=list(inputs), output_names=OUTPUTS)
    for fp16 in [False, True]:
        engine_path = os.path.join(out_dir, 'source_net_it%d_%s.engine' % (arg.iters, 'fp16' if fp16 else 'fp32'))
        t = time.time()
        build_engine(onnx_path, engine_path, fp16)
        trt_net = TrtSourceNet(engine_path)
        t_trt = timeit(lambda: trt_net(inputs))
        img = render(trt_net(inputs))
        print('TensorRT %s: network %.1f ms + render %.1f ms = %.1f ms (%.1f FPS), PSNR vs GT %.2f, vs PyTorch %.2f '
              '(engine built in %.0f s: %s)' % ('fp16' if fp16 else 'fp32', t_trt, t_render, t_trt + t_render,
                                                1000 / (t_trt + t_render), psnr(img, gt), psnr(img, ref), time.time() - t, engine_path), flush=True)
