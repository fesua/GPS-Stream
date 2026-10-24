import time
import torch
import torch.nn.functional as F
from lib.temporal_gps import TemporalGPS
from lib.patch_step3 import PatchStep3
from export_trt import source_inputs
from export_trt_patch import TrtSourceNetF3

# Step 3 with the per-frame network as a TensorRT engine: the engine computes the whole frame (xyz, rot, scale, opacity,
# depth, f3); STSS (PyTorch, 1/8 map) marks the cells, the P x P tiles around them take the engine's Gaussian maps, the
# rest is kept. = the dense emulation (PatchStep3) at TensorRT cost.


class PatchTRT:
    def __init__(self, model: TemporalGPS, engine_path, patch=128, full=False):
        self.m, self.c, self.full = model, patch // 8, full   # full: take the engine's output everywhere (= the per-frame model as an engine)
        self.net = TrtSourceNetF3(engine_path)

    def _run(self, data):
        xyz, rot, scale, opacity, depth, f3 = self.net(source_inputs(data))
        inp = self.m.inputs(data)
        g = {'xyz': xyz.clone(), 'rot': rot.clone(), 'scale': scale.clone(), 'opacity': opacity.clone(), 'depth': depth.clone(), 'img': inp['image'],
             'valid': (inp['mask'][:, :1] > 0.5).flatten(1)}
        return inp, f3.clone(), g

    @torch.no_grad()
    def keyframe(self, data):
        inp, f3, g = self._run(data)
        return {'f3': f3, 'image': inp['image'], 'g': g, 'flow': None}

    @torch.no_grad()
    def step(self, data, state):
        T = {}; tick = lambda: (torch.cuda.synchronize(), time.perf_counter())[1]
        t0 = tick()
        inp, f3, gn = self._run(data)
        T['engine'] = tick() - t0; t0 = tick()
        changed = (torch.sigmoid(self.m.stss(f3, state['f3'])) > 0.5).float()
        M = torch.ones_like(changed, dtype=torch.bool) if self.full else PatchStep3.footprint_mask(changed, self.c, 'tile', 0)
        Mf = F.interpolate(M.float(), scale_factor=8, mode='nearest') > 0.5
        gp = state['g']
        g = {'xyz': torch.where(Mf.flatten(1)[..., None], gn['xyz'], gp['xyz']), 'valid': gn['valid']}
        for k in ('rot', 'scale', 'opacity', 'img', 'depth'):
            g[k] = torch.where(Mf, gn[k], gp[k])
        T['stss+paste'] = tick() - t0
        new_state = {'f3': torch.where(M, f3, state['f3']), 'image': torch.where(Mf, inp['image'], state['image']), 'g': g, 'flow': None}
        return {'g': g, 'flow': None, 'mask': M.float(), 'changed': changed, 'times': T}, new_state
