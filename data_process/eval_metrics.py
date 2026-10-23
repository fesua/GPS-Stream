import os, sys, glob, json, argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import lpips
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.gs_utils.loss_utils import ssim
from step_dynerf_test import work_sets

# python data_process/eval_metrics.py cook_spinach cut_roasted_beef flame_steak sear_steak --exp gps_plus_v2 --frames 220 300 --masked
# PSNR / SSIM (3DGS gaussian window) / LPIPS (alex) of test.py renders against the novel-view GT written by
# step_dynerf_test.py.
# --masked: valid region only (renders from test.py --save_mask), following ENeRF / GPS-Gaussian+: outside the
# mask is zeroed in both images, PSNR averages over valid pixels, SSIM / LPIPS use the mask's bounding box.
# --person_box: additionally restrict to the person's bounding box in the GT view (COCO Faster R-CNN), like
# ENeRF's evaluation of human-centric data. Boxes are cached in GPS_plus_data/test/<seq>_process/person_box.json.


def person_boxes(items):
    import re
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    model, boxes = None, {}
    for seq in sorted({it[0] for it in items}):
        cache = 'GPS_plus_data/test/%s_process/person_box.json' % seq
        boxes[seq] = json.load(open(cache)) if os.path.exists(cache) else {}
        todo = [it[4] for it in items if it[0] == seq and it[4] not in boxes[seq]]
        for gt in todo:
            if model is None:
                model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
            img = torch.from_numpy(cv2.imread(gt)[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda() / 255
            with torch.no_grad():
                o = model([img])[0]
            person = o['boxes'][o['labels'] == 1]  # sorted by score
            boxes[seq][gt] = person[0].round().int().tolist() if len(person) else None
        if todo:
            json.dump(boxes[seq], open(cache, 'w'))
    return {gt: b for d in boxes.values() for gt, b in d.items()}


class Pairs(Dataset):
    def __init__(self, items, masked, boxes=None):
        self.items, self.masked, self.boxes = items, masked, boxes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pred, gt = self.items[i][-2:]
        load = lambda f: torch.from_numpy(cv2.imread(f)[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
        mask = torch.ones(1, 1024, 1024)
        if self.masked:
            mask = torch.from_numpy(cv2.imread(pred[:-4] + '_mask.png', 0) > 127).float()[None]
        if self.boxes is not None:
            x0, y0, x1, y1 = self.boxes[gt]
            box = torch.zeros_like(mask)
            box[:, y0:y1, x0:x1] = 1
            mask = mask * box
        return load(pred), load(gt), mask, i


def collect(exp, seq, view, frames):
    # a scene missing a novel camera got the other novel view in its place (step_dynerf_test.available_set)
    calib = json.load(open('GPS_plus_data/test/%s_process/calibration_full.json' % seq))
    missing = {'s%d' % (s_i + 1) for s_i, cams in enumerate(work_sets) if cams[view - 2] not in calib}
    out = []
    for f in sorted(glob.glob('experiments/%s/show_%s_process/*_%02d.jpg' % (exp, seq, view))):
        name = os.path.basename(f)[:-7]
        work_set, fr = name.split('_')[-2], int(name.split('_')[-1])
        if work_set in missing or not frames[0] <= fr < frames[1]:
            continue
        gt = 'GPS_plus_data/test/%s_process/img/%s/%d.jpg' % (seq, name, view)
        out.append((seq, view, work_set, f, gt))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--exp', default='gps_plus')
    parser.add_argument('--frames', type=int, nargs=2, default=[0, 100000], help='evaluate frames [start, end)')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--person_box', action='store_true')
    parser.add_argument('--only', type=str, default=None, help='e.g. s2:3 to evaluate one novel view (cam00)')
    parser.add_argument('--tag', default='', help='output name suffix')
    arg = parser.parse_args()
    out = 'experiments/%s/metrics_dynerf%s' % (arg.exp, arg.tag)

    items = []
    for seq in arg.scenes:
        for view in [2, 3]:
            items += collect(arg.exp, seq, view, arg.frames)
    if arg.only:
        ws, v = arg.only.split(':')
        items = [it for it in items if it[2] == ws and it[1] == int(v)]
    boxes = person_boxes(items) if arg.person_box else None
    if boxes is not None:
        items = [it for it in items if boxes[it[4]] is not None]
    print('pairs:', len(items))

    lp_alex = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    loader = DataLoader(Pairs(items, arg.masked, boxes), batch_size=1, num_workers=8)

    rows = []
    with torch.no_grad():
        for b, (pred, gt, mask, idx) in enumerate(loader):
            pred, gt, mask = pred.cuda(), gt.cuda(), mask.cuda()
            if arg.masked or boxes is not None:
                pred, gt = pred * mask, gt * mask
                mse = ((pred - gt) ** 2).sum() / (mask.sum() * 3)
                ys, xs = torch.nonzero(mask[0, 0], as_tuple=True)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                pred, gt = pred[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1]
            else:
                mse = ((pred - gt) ** 2).mean()
            psnr = 10 * torch.log10(1 / mse)
            s = ssim(pred, gt)
            la = lp_alex(pred * 2 - 1, gt * 2 - 1).flatten()[0]
            i = idx.item()
            seq, view, ws = items[i][:3]
            rows.append((seq, view, ws, os.path.basename(items[i][3]),
                         psnr.item(), s.item(), la.item(), mask.mean().item()))
            if b % 1000 == 0:
                print('%d / %d' % (len(rows), len(items)), flush=True)

    with open(out + '_per_frame.csv', 'w') as f:
        f.write('scene,view,set,file,psnr,ssim,lpips,valid\n')
        for r in rows:
            f.write('%s,%d,%s,%s,%.4f,%.5f,%.5f,%.4f\n' % r)

    groups = defaultdict(list)
    for r in rows:
        m = np.array(r[4:])
        groups[(r[0], 'all', 'all')].append(m)
        groups[(r[0], 'v%d' % r[1], 'all')].append(m)
        groups[(r[0], 'all', r[2])].append(m)
    # scene average (flame_salmon_1 has 4x more frames, don't let it dominate)
    groups[('MEAN', 'all', 'all')] = [np.mean(v, 0) for k, v in groups.items() if k[1:] == ('all', 'all')]
    region = ('valid region' if arg.masked else 'full frame') + (' & person box' if arg.person_box else '')
    lines = ['frames [%d, %d)  %s%s' % (*arg.frames, region, '  view ' + arg.only if arg.only else ''),
             '%-18s %-4s %-4s %6s %7s %7s %7s %6s' % ('scene', 'view', 'set', 'n', 'PSNR', 'SSIM', 'LPIPS', 'valid')]
    for key in sorted(groups, key=lambda k: (k[0] == 'MEAN', k)):
        m = np.mean(groups[key], 0)
        lines.append('%-18s %-4s %-4s %6d %7.2f %7.4f %7.4f %6.3f' % (*key, len(groups[key]), *m))
    open(out + '_summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
