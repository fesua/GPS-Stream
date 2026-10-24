import os
import time
import argparse
import numpy as np
import torch
import tensorrt as trt
from config.stereo_human_config import ConfigStereoHuman as config
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from export_trt import SourceNet, explicit_group_norm, build_engine, source_inputs, timeit, depth2pc_export, LOG

# Per-frame network for the PATCH pipeline as one TensorRT engine: the SourceNet outputs (xyz, rot, scale, opacity) plus
# the depth map and the encoder's 1/8 features (STSS input). python export_trt_patch.py --ckpt <per-frame ckpt>
OUTPUTS_F3 = ['xyz', 'rot', 'scale', 'opacity', 'depth', 'f3']


class SourceNetF3(SourceNet):
    def forward(self, image, flow_init, intr, ref_intr, extr, tf_x):
        m = self.m
        img_feat = m.img_encoder(image)
        f3 = img_feat[2]
        c0, c1 = m.loftr_coarse(*f3.split(1), None, None)
        feat_cs = torch.cat((c0, c1), 0)
        flow = torch.cat(m.raft_stereo(feat_cs, flow_init=flow_init, iters=self.iters, test_mode=True), 0)
        offset = (ref_intr[:, 0, 2] - intr[:, 0, 2])[:, None, None, None]
        depth = -(offset - flow) / tf_x[:, None, None, None]
        rot, scale, opacity, res_depth = m.gs_parm_regresser(image, depth, (img_feat[0], img_feat[1], feat_cs))
        d = depth + res_depth
        return depth2pc_export(d, extr, intr), rot, scale, opacity, d, f3


class TrtSourceNetF3:
    def __init__(self, engine_path):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(open(engine_path, 'rb').read())
        self.ctx = self.engine.create_execution_context()
        self.outs = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), device='cuda') for n in OUTPUTS_F3}
        for n, t in self.outs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())

    def __call__(self, inputs):
        for n, t in inputs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return [self.outs[n] for n in OUTPUTS_F3]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', default='experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth'); p.add_argument('--iters', type=int, default=8)
    p.add_argument('--sample', default='cook_spinach_s2_0250'); p.add_argument('--precisions', nargs='+', default=['fp16', 'fp32']); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(a.ckpt)), 'trt'); os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, 'source_net_f3_it%d.onnx' % a.iters)
    cfg = config(); cfg.load('config/stage.yaml'); cfg = cfg.get_cfg(); cfg.defrost(); cfg.raft.val_iters = a.iters; cfg.dataset.inverse_depth_init = 0.5
    model = RtStereoHumanModel(cfg, with_gs_render=True).cuda(); model.load_state_dict(torch.load(a.ckpt, map_location='cuda')['network']); model.eval(); model.raft_stereo.freeze_bn()
    seq = a.sample.rsplit('_', 2)[0] + '_process'
    ds = StereoHumanDataset(cfg.dataset, phase=seq)
    data = torch.utils.data.default_collate([ds.get_item(ds.sample_list.index(a.sample), novel_id=[2])])
    for view in ['lmain', 'rmain']:
        data[view] = {k: v.cuda() for k, v in data[view].items()}
    inputs = source_inputs(data)
    net = SourceNetF3(model, a.iters).eval()
    with torch.no_grad():
        ref = net(*inputs.values()); t_net = timeit(lambda: net(*inputs.values()))
    print('PyTorch fp32 network (with depth + f3 outputs): %.1f ms' % t_net, flush=True)
    with torch.no_grad():
        torch.onnx.export(SourceNetF3(explicit_group_norm(model), a.iters).eval(), tuple(inputs.values()), onnx_path, opset_version=17, input_names=list(inputs), output_names=OUTPUTS_F3)
    for prec in a.precisions:
        engine_path = os.path.join(out_dir, 'source_net_f3_it%d_%s.engine' % (a.iters, prec))
        t0 = time.time(); build_engine(onnx_path, engine_path, prec == 'fp16')
        trt_net = TrtSourceNetF3(engine_path)
        outs = trt_net(inputs); t_trt = timeit(lambda: trt_net(inputs))
        tf = inputs['tf_x'].abs()[0]
        print('TensorRT %s: network %.1f ms | vs PyTorch: depth-as-disparity mean |diff| %.3f px, f3 mean |diff| %.4f, opacity %.4f (engine built in %.0f s: %s)' % (
            prec, t_trt, ((outs[4] - ref[4]).abs() * tf).mean().item(), (outs[5] - ref[5]).abs().mean().item(), (outs[3] - ref[3]).abs().mean().item(), time.time() - t0, engine_path), flush=True)
