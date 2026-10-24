# Training

GPS-Stream has three trained parts:

1. the **per-frame network**: GPS-Gaussian+ fine-tuned on DyNeRF;
2. **STSS**, the change selector that picks the tiles to recompute;
3. the **tile-condition weights**: the per-frame network fine-tuned under GPS-Stream's execution condition (128 px tiles, 4-cell border).

The released weights cover all three for DyNeRF. This page shows how to reproduce them, or how to adapt them to new scenes.
Run everything from the repository root with `export PYTHONPATH=.`.

## Data

Place the [DyNeRF (Neural 3D Video)](https://github.com/facebookresearch/Neural_3D_Video) scenes in `GPS_plus_data/raw_dynerf/<scene>`, then:

```bash
cd data_process
# evaluation scenes (one per call): GPS_plus_data/test/<scene>_process
python step_dynerf_test.py -i cook_spinach
# per-frame network training: GPS_plus_data/{train,val} (frames 0-219, stride 2)
python step_dynerf_train.py -i cook_spinach cut_roasted_beef flame_steak sear_steak
# consecutive frames for STSS and the tile-condition fine-tuning: GPS_plus_data/temporal/{train,val}
python step_dynerf_train.py -i cook_spinach cut_roasted_beef flame_steak sear_steak --stride 1 --root ../GPS_plus_data/temporal/
cd ..
```

Evaluation uses frames 220-299 of each sequence: frame 220 is the start frame, frames 221-299 are scored against novel views 2 and 3.

For another camera rig, prepare the data as in the original GPS-Gaussian+ ([docs/GPS-Gaussian-plus_README.md](GPS-Gaussian-plus_README.md)).
The streaming scripts set `inverse_depth_init = 0.5`, which suits DyNeRF scenes about 1.5-2 m from the cameras; change it for scenes at another distance.

## 1. Per-frame network

Fine-tune the GPS-Gaussian+ weights on DyNeRF in three stages. Each config's `stage1_ckpt` points to the previous stage's result;
set the first one (`dynerf_ft.yaml`) to the downloaded GPS-Gaussian+ checkpoint.

```bash
python train.py --cfg config/dynerf_ft.yaml     # 40k steps, lr 2e-4
python train.py --cfg config/dynerf_it8.yaml    # 8 GRU iterations in training
python train.py --cfg config/dynerf_jit2.yaml   # + inverse-depth init jitter 0.3
```

All three use chamfer weight 0.005 (DyNeRF's depth range is larger than the original data's). The streaming scripts expect the result at
`experiments/gps_plus_dynerf_jit2_1007/ckpt/gps_plus_dynerf_jit2_final.pth` (constant `TEACHER` in the scripts).

## 2. STSS

STSS compares each 1/8 feature cell with the features of the frame where that cell was last updated, and marks the cells that changed.
It is trained inside the patch pipeline against the image-difference target (cell colour change above 8/255):

```bash
python train_stss_patch.py --exp stss_patch64 --patch 64 --steps 4000 --init <TemporalGPS checkpoint>
# -> experiments/stss_patch64/ckpt/step4000.pth
```

`--init` is the starting point: a TemporalGPS checkpoint from `train_temporal.py` (default `experiments/temporal_r8_ref/ckpt/step8000.pth`).

## 3. Tile-condition fine-tuning

A tile crop sees zero padding beyond its border and the start frame's normalisation statistics, so GPS-Stream needs weights trained under
that condition. Without this step the per-frame weights lose about 1.2 dB on tile crops (PyTorch: 27.96 dB untrained, 29.17 dB trained).

```bash
python train_patch_sparse.py --exp patch_sparse_ft128h4_rc4 --patch 128 --halo 4 --reg_crop 4 --steps 8000
```

8000 steps, lr 2e-5, clips of 2-4 frames, encoder frozen; about 1.5 h on an RTX 3090.
The first frame of each clip and the maps outside the tiles come from the frozen per-frame network, as at inference.

## 4. Engines, timing and evaluation

```bash
# encoder engine
python bench_trt_sparse.py --build --only enc
# crop and tile-row attention engines from the fine-tuned weights
python build_sparse_trt.py --teacher experiments/patch_sparse_ft128h4_rc4/ckpt/step8000.pth \
    --out trt/sparse_ft128h4_rc4 --patch 128 --halo 4 --reg_margin 4
# per-frame time, TensorRT and PyTorch (cook_spinach s2, frames 221-260)
python bench_patch_sparse_trt.py --teacher experiments/patch_sparse_ft128h4_rc4/ckpt/step8000.pth \
    --engine_dir experiments/gps_plus_dynerf_jit2_1007/trt/sparse_ft128h4_rc4 --patch 128 --halo 4 --reg_margin 4
# full-protocol renders (all four scenes, sequences 1-3)
python test_patch3.py --exp gps_stream --mask stss --ckpt experiments/stss_patch64/ckpt/step4000.pth \
    --patch 128 --exec sparse_trt --halo 4 --reg_cells 4 --reg_mode crop \
    --engine experiments/gps_plus_dynerf_jit2_1007/trt/sparse_ft128h4_rc4 --teacher experiments/patch_sparse_ft128h4_rc4/ckpt/step8000.pth
# metrics against the novel-view ground truth: image quality, then temporal consistency
python data_process/eval_metrics.py cook_spinach cut_roasted_beef flame_steak sear_steak --exp gps_stream --frames 221 300 --masked --tag _t221
python data_process/eval_metrics.py cook_spinach cut_roasted_beef flame_steak sear_steak --exp gps_stream --frames 221 300 --masked --person_box --tag _t221_person
python data_process/eval_temporal.py cook_spinach cut_roasted_beef flame_steak sear_steak --exps gps_stream
```

The per-frame reference: `python export_trt_patch.py` builds the whole network as one engine; render it with `test_patch3.py --exec trt_full`
(TensorRT) or `--exec perframe` without `--teacher` (PyTorch).
