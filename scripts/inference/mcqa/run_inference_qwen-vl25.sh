#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/mcqa/qwen-vl25-7b.json"

python inference.py \
    --model "qwen-vl25" \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --task "mcqa" \
    --num_frames 8 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
