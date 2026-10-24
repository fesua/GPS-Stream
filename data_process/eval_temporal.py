import os, sys, argparse
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import lpips

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import collect

# Temporal consistency (flicker) of novel-view renders against the ground truth, on the same frames as eval_metrics.py.
#   tOF  = mean over pixels of || OF(R_t-1, R_t) - OF(G_t-1, G_t) ||_1   (TecoGAN, Chu et al. 2020; Farneback flow, px)
#   tLP  = | LPIPS(R_t-1, R_t) - LPIPS(G_t-1, G_t) |                    (TecoGAN; AlexNet LPIPS, reported x100)
#   static flicker = on pixels where the GT does not change (|G_t - G_t-1| < 3, 0-255, mean over RGB), mean |R_t - R_t-1|;
#                    the same quantity on the GT is its noise floor.
# Lower is better for all three. Pixels: the valid region shared by every evaluated experiment at t-1 and t (render masks).
# Run from the folder that holds experiments/<exp> and GPS_plus_data/test (like eval_metrics.py):
# python data_process/eval_temporal.py cook_spinach cut_roasted_beef flame_steak sear_steak --exps perframe_i8 patch3_sparse_trt_p128_h4_rc4_att

FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def sequences(exps, scenes, frames):
    """[(scene, view, work_set, [(frame, gt, {exp: render})...])] with every experiment present, frames sorted"""
    out = []
    for sc in scenes:
        for v in (2, 3):
            per = {e: {(ws, int(os.path.basename(f)[:-7].split('_')[-1])): (f, gt) for _, _, ws, f, gt in collect(e, sc, v, frames)} for e in exps}
            keys = sorted(set.intersection(*[set(p) for p in per.values()]))
            for ws in sorted({k[0] for k in keys}):
                fr = [(t, per[exps[0]][(w, t)][1], {e: per[e][(w, t)][0] for e in exps}) for w, t in keys if w == ws]
                out.append((sc, v, ws, fr))
    return out


def flow_pass(job):
    """tOF and static flicker of one sequence (CPU)"""
    sc, v, ws, fr, exps = job
    load = lambda f: cv2.imread(f)
    mask = lambda f: cv2.imread(f[:-4] + '_mask.png', 0) > 127
    res = []
    prev = None
    for t, gt_f, rf in fr:
        cur = {'gt': load(gt_f)}
        for e in exps:
            cur[e] = load(rf[e]); cur[e + '_m'] = mask(rf[e])
        if prev is not None and t - 1 == prev['t']:
            m = np.ones(cur['gt'].shape[:2], bool)
            for e in exps:
                m &= cur[e + '_m'] & prev[e + '_m']
            g0, g1 = prev['gt'].astype(np.float32), cur['gt'].astype(np.float32)
            of_gt = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev['gt'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur['gt'], cv2.COLOR_BGR2GRAY), None, **FB)
            static = (np.abs(g1 - g0).mean(2) < 3) & m
            row = {'t': t, 'gt_floor': np.abs(g1 - g0).mean(2)[static].mean() if static.any() else np.nan}
            for e in exps:
                of = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev[e], cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur[e], cv2.COLOR_BGR2GRAY), None, **FB)
                row[e + '_tOF'] = np.abs(of - of_gt).sum(2)[m].mean()
                row[e + '_static'] = np.abs(cur[e].astype(np.float32) - prev[e].astype(np.float32)).mean(2)[static].mean() if static.any() else np.nan
            res.append(row)
        cur['t'] = t
        prev = cur
    return sc, v, ws, res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('scenes', nargs='+'); p.add_argument('--exps', nargs='+', required=True)
    p.add_argument('--frames', type=int, nargs=2, default=[220, 300], help='renders read; pairs (t-1, t) with both in range')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    seqs = sequences(a.exps, a.scenes, a.frames)
    with Pool(a.workers) as pool:
        flows = pool.map(flow_pass, [(sc, v, ws, fr, a.exps) for sc, v, ws, fr in seqs])
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    ten = lambda img, m: torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda()[None] / 255 * torch.from_numpy(m).cuda()[None, None]
    stats = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for (sc, v, ws, fr), (_, _, _, rows) in zip(seqs, flows):
            byt = {t: (gt_f, rf) for t, gt_f, rf in fr}
            for row in rows:
                t = row['t']; (g0f, r0), (g1f, r1) = byt[t - 1], byt[t]
                m = np.ones((1024, 1024), bool)
                for e in a.exps:
                    for f in (r0[e], r1[e]):
                        m &= cv2.imread(f[:-4] + '_mask.png', 0) > 127
                lp_pair = lambda f0, f1: lp(ten(cv2.imread(f0), m) * 2 - 1, ten(cv2.imread(f1), m) * 2 - 1).item()
                lg = lp_pair(g0f, g1f)
                stats[sc]['gt_floor'].append(row['gt_floor'])
                for e in a.exps:
                    stats[sc][e + '_tOF'].append(row[e + '_tOF'])
                    stats[sc][e + '_tLP'].append(100 * abs(lp_pair(r0[e], r1[e]) - lg))
                    stats[sc][e + '_static'].append(row[e + '_static'])
    print('temporal consistency, %d sequences x views, frame pairs (t-1, t) within %d-%d; lower is better' % (len(seqs), a.frames[0], a.frames[1] - 1))
    print('%-40s %-17s %8s %10s %16s' % ('experiment', 'scene', 'tOF', 'tLP x100', 'static flicker'))
    allm = defaultdict(list)
    for sc in a.scenes:
        for e in a.exps:
            vals = [np.nanmean(stats[sc][e + k]) for k in ('_tOF', '_tLP', '_static')]
            for k, val in zip(('tOF', 'tLP', 'static'), vals): allm[(e, k)].append(val)
            print('%-40s %-17s %8.3f %10.3f %16.3f' % (e, sc, *vals))
        allm[('gt', 'static')].append(np.nanmean(stats[sc]['gt_floor']))
    for e in a.exps:
        print('%-40s %-17s %8.3f %10.3f %16.3f' % (e, 'MEAN (4 scenes)', *[np.mean(allm[(e, k)]) for k in ('tOF', 'tLP', 'static')]))
    print('%-40s %-17s %8s %10s %16.3f' % ('GT (noise floor)', 'MEAN (4 scenes)', '', '', np.mean(allm[('gt', 'static')])))
