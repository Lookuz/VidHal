#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/mcqa/moviechat.json"

python inference.py \
    --model "moviechat" \
    --config_path "models/MovieChat/configs/eval.yaml" \
    --task "mcqa" \
    --num_frames 8 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
