#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/mcqa/intern-vl25-8b.json"

python inference.py \
    --model "intern-vl25" \
    --model_path "OpenGVLab/InternVL2_5-8B" \
    --task "mcqa" \
    --num_frames 16 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
