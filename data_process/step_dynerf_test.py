import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from step_0rect import get_rectified_stereo_data

# python step_dynerf_test.py -i cook_spinach
# DyNeRF (Neural 3D Video) scene -> GPS-Gaussian+ test format (processed_data_root/test/<scene>_process).
# Calibration follows dynerf0.py (LLFF poses_bounds.npy -> OpenCV w2c).

data_root = '../GPS_plus_data/raw_dynerf/'
processed_data_root = '../GPS_plus_data/test/'

# Bottom row of the rig, left -> right: cam10 09 08 07 06 00 05 04 03 02 01.
# Each work set: [novel 2, novel 3, source 0 (left), source 1 (right)], same layout as step_1.py.
work_sets = [
    ['cam09', 'cam08', 'cam10', 'cam07'],
    ['cam06', 'cam00', 'cam07', 'cam05'],
    ['cam04', 'cam03', 'cam05', 'cam02'],
]

RS = 1024


def load_calib(scene_dir, scale):
    cams = sorted([f[:-4] for f in os.listdir(scene_dir) if f.endswith('.mp4')])
    pose_bounds = np.load(os.path.join(scene_dir, 'poses_bounds.npy'))
    pose = pose_bounds[:, :-2].reshape(-1, 3, 5)
    assert len(cams) == pose.shape[0]

    calib = {}
    for i, cam in enumerate(cams):
        R, T, (H, W, F) = pose[i, :, :3], pose[i, :, 3:4], pose[i, :, 4]
        # LLFF [down, right, back] -> OpenCV [right, down, forward]
        c2w = np.eye(4)
        c2w[:3, :3] = np.concatenate([R[:, 1:2], R[:, 0:1], -R[:, 2:3]], -1)
        c2w[:3, 3:4] = T * scale
        w2c = np.linalg.inv(c2w)
        K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
        calib[cam] = {
            'K': K.tolist(),
            'R': w2c[:3, :3].tolist(),
            'T': w2c[:3, 3:4].tolist(),
            'distCoeff': np.zeros(5).tolist(),
            'imgSize': [int(W), int(H)],
        }
    return calib


def crop_square(img, x, H):
    # landscape: keep the H x H columns starting at x
    return cv2.resize(img[:, x:x + H], (RS, RS))


def square_intr(K, x, H):
    K = np.array(K, dtype=float).copy()
    K[0, 2] -= x
    K[:2] *= RS / H
    return K


def available_set(cam_list, calib):
    # some scenes miss a camera (e.g. cam03 in coffee_martini): reuse the other novel view
    novel = [c for c in cam_list[:2] if c in calib]
    assert all(c in calib for c in cam_list[2:]) and len(novel) > 0, cam_list
    return [novel[0], novel[-1]] + cam_list[2:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='dynerf scene name')
    parser.add_argument('-n', '--num_frames', type=int, default=0, help='0: all frames')
    parser.add_argument('-s', '--scale', type=float, default=0.1, help='poses_bounds unit -> meter')
    # rectified source pairs are ~410 px apart (1024 px crop) when both are center-cropped, far beyond the
    # ~100 px the model was trained on; cropping the right view further right brings the pair closer
    parser.add_argument('--shift', type=int, default=300, help='crop x of source 1 - crop x of source 0 (full-res px)')
    arg = parser.parse_args()

    scene_dir = os.path.join(data_root, arg.input)
    out_root = os.path.join(processed_data_root, arg.input + '_process')
    calib = load_calib(scene_dir, arg.scale)
    Path(out_root).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(out_root, 'calibration_full.json'), 'w') as f:
        json.dump(calib, f, indent=4)

    for s_i, cam_list in enumerate(work_sets):
        scene_n = 's%d' % (s_i + 1)
        cam_list = available_set(cam_list, calib)
        print(scene_n, cam_list)
        caps = {cam: cv2.VideoCapture(os.path.join(scene_dir, cam + '.mp4')) for cam in set(cam_list)}
        num = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
        if arg.num_frames > 0:
            num = min(num, arg.num_frames)

        W, H = calib[cam_list[2]]['imgSize']
        x_c = (W - H) // 2
        x_0 = int(np.clip(x_c - arg.shift // 2, 0, W - H))
        x_1 = int(np.clip(x_0 + arg.shift, 0, W - H))
        for t in tqdm(range(num), desc=scene_n):
            frames = {}
            for cam, cap in caps.items():
                ok, frames[cam] = cap.read()
                assert ok, (cam, t)

            def view(cam):
                c = calib[cam]
                extr = np.concatenate([np.array(c['R']), np.array(c['T'])], 1)
                return frames[cam], np.array(c['K']), extr

            name = '%s_%s_%04d' % (arg.input, scene_n, t)
            img_dir = os.path.join(out_root, 'img', name)
            msk_dir = os.path.join(out_root, 'mask', name)
            par_dir = os.path.join(out_root, 'parameter', name)
            for d in [img_dir, msk_dir, par_dir]:
                Path(d).mkdir(exist_ok=True, parents=True)

            # source views 0, 1 (rectified)
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

            # novel views 2-5 (original, unrectified)
            for cam_i, cam in enumerate(cam_list):
                img, K, extr = view(cam)
                cv2.imwrite(os.path.join(img_dir, '%d.jpg' % (cam_i + 2)), crop_square(img, x_c, H))
                np.save(os.path.join(par_dir, '%d_extrinsic.npy' % (cam_i + 2)), extr)
                np.save(os.path.join(par_dir, '%d_intrinsic.npy' % (cam_i + 2)), square_intr(K, x_c, H))
