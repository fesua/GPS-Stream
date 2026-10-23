import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from step_0rect import get_rectified_stereo_data
from step_dynerf_test import load_calib, crop_square, square_intr, data_root, RS

# python step_dynerf_train.py -i cook_spinach cut_roasted_beef flame_steak sear_steak
# DyNeRF scenes -> GPS-Gaussian+ train/val format (GPS_plus_data/{train,val}).
# Paper split (GPS-Gaussian+ Sec. V-A): the first 220 frames are training data, frames 220-299 are left for
# testing with step_dynerf_test.py. Val frames are odd frames inside the training range (train uses even ones).

processed_data_root = '../GPS_plus_data/'

# Rig rows, left -> right. Work set: sources 3 cameras apart (like the test sets), the 2 in between are
# novel views 2, 3; the sources themselves (unrectified) are novel views 4, 5.
rows = {
    'b': ['cam10', 'cam09', 'cam08', 'cam07', 'cam06', 'cam00', 'cam05', 'cam04', 'cam03', 'cam02', 'cam01'],
    't': ['cam20', 'cam19', 'cam18', 'cam17', 'cam16', 'cam15', 'cam14', 'cam13', 'cam12', 'cam11'],
}


def center(c):
    return -np.array(c['R']).T @ np.array(c['T']).reshape(3)


def extra_cams(calib, cam_list, n):
    # the n cameras closest to the source pair (mostly the other row), as additional supervision views
    mid = (center(calib[cam_list[2]]) + center(calib[cam_list[3]])) / 2
    rest = [c for c in calib if c not in cam_list]
    return sorted(rest, key=lambda c: np.linalg.norm(center(calib[c]) - mid))[:n]


def work_sets(calib):
    sets = []
    for r, row in rows.items():
        for i in range(len(row) - 3):
            src = [row[i], row[i + 3]]
            novel = [c for c in row[i + 1:i + 3] if c in calib]
            if all(c in calib for c in src) and novel:
                sets.append(('%s%d' % (r, i), [novel[0], novel[-1]] + src))
    return sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='dynerf scene names')
    parser.add_argument('--train_end', type=int, default=220, help='frames [0, train_end) are training data')
    parser.add_argument('--stride', type=int, default=2, help='train frame stride')
    parser.add_argument('-s', '--scale', type=float, default=0.1)
    parser.add_argument('--shift', type=int, default=300)
    parser.add_argument('--extra_views', type=int, default=0, help='extra supervision views 6, 7, ... per sample')
    parser.add_argument('--root', type=str, default=processed_data_root, help='output: <root>/{train,val}')
    arg = parser.parse_args()

    for scene in arg.input:
        scene_dir = os.path.join(data_root, scene)
        calib = load_calib(scene_dir, arg.scale)
        for s_i, (set_n, cam_list) in enumerate(work_sets(calib)):
            cam_list = cam_list + extra_cams(calib, cam_list, arg.extra_views)
            caps = {cam: cv2.VideoCapture(os.path.join(scene_dir, cam + '.mp4')) for cam in set(cam_list)}
            phase = {t: 'train' for t in range(0, arg.train_end, arg.stride)}
            # one odd (never trained) frame per work set, spread over the training range
            phase[(37 + 74 * s_i) % arg.train_end | 1] = 'val'

            W, H = calib[cam_list[2]]['imgSize']
            x_c = (W - H) // 2
            x_0 = int(np.clip(x_c - arg.shift // 2, 0, W - H))
            x_1 = int(np.clip(x_0 + arg.shift, 0, W - H))
            for t in tqdm(range(max(phase) + 1), desc='%s %s %s' % (scene, set_n, cam_list)):
                if t not in phase:
                    for cap in caps.values():
                        cap.grab()
                    continue
                frames = {}
                for cam, cap in caps.items():
                    ok, frames[cam] = cap.read()
                    assert ok, (cam, t)

                def view(cam):
                    c = calib[cam]
                    extr = np.concatenate([np.array(c['R']), np.array(c['T'])], 1)
                    return frames[cam], np.array(c['K']), extr

                out_root = os.path.join(arg.root, phase[t])
                name = '%s_%s_%04d' % (scene, set_n, t)
                img_dir = os.path.join(out_root, 'img', name)
                msk_dir = os.path.join(out_root, 'mask', name)
                par_dir = os.path.join(out_root, 'parameter', name)
                for d in [img_dir, msk_dir, par_dir]:
                    Path(d).mkdir(exist_ok=True, parents=True)

                rect = get_rectified_stereo_data(view(cam_list[2]), view(cam_list[3]), (W, H))
                for k, x in [(0, x_0), (1, x_1)]:
                    cv2.imwrite(os.path.join(img_dir, '%d.jpg' % k), crop_square(rect['img%d' % k], x, H))
                    cv2.imwrite(os.path.join(msk_dir, '%d.jpg' % k), crop_square(rect['mask%d' % k], x, H))
                parm = rect['camera']
                parm['intr0'] = square_intr(parm['intr0'], x_0, H)
                parm['intr1'] = square_intr(parm['intr1'], x_1, H)
                parm['Tf_x'] = parm['Tf_x'] * RS / H
                with open(os.path.join(par_dir, '0_1.json'), 'w') as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in parm.items()}, f, indent=1)

                for cam_i, cam in enumerate(cam_list):
                    img, K, extr = view(cam)
                    cv2.imwrite(os.path.join(img_dir, '%d.jpg' % (cam_i + 2)), crop_square(img, x_c, H))
                    np.save(os.path.join(par_dir, '%d_extrinsic.npy' % (cam_i + 2)), extr)
                    np.save(os.path.join(par_dir, '%d_intrinsic.npy' % (cam_i + 2)), square_intr(K, x_c, H))
