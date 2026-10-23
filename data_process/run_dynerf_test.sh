#!/bin/bash
# DyNeRF (N3V) scene: process -> novel view test (view 2, 3) -> free view video
# usage (from repo root): bash data_process/run_dynerf_test.sh cook_spinach
# metrics afterwards: python data_process/eval_metrics.py <scenes...> --exp $EXP
set -e
SCENE=$1
PY=${PY:-python}
EXP=${EXP:-gps_plus_v2}
# DyNeRF scenes sit ~1.5 m from the rig (pose scale 0.1), much closer than the default init (0.3 -> 3.3 m)
INV=${INV:-0.5}
SHIFT=${SHIFT:-300}
# the conda env's ffmpeg has libx264 (the base anaconda one does not)
FFMPEG=${FFMPEG:-$(dirname $(command -v $PY))/ffmpeg}
[ -x "$FFMPEG" ] || FFMPEG=ffmpeg

cd data_process
$PY step_dynerf_test.py -i $SCENE --shift $SHIFT
cd ..

for V in 2 3; do
    $PY test.py -i $SCENE -v $V --exp $EXP --inv_depth $INV
done

N=$(ls GPS_plus_data/test/${SCENE}_process/img | grep -c "_s1_")
$PY run_interpolation.py -i ${SCENE} --cams cam10 cam07 cam05 cam02 --end_frame $N --exp $EXP --inv_depth $INV
$FFMPEG -y -loglevel error -framerate 30 -i experiments/$EXP/show_free_${SCENE}/%03d.jpg \
    -c:v libx264 -pix_fmt yuv420p -crf 18 experiments/$EXP/free_view_${SCENE}.mp4
