#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/mcqa/videochat2.json"
config_path="models/VideoChat2/configs/config.json"

python inference.py \
    --model "videochat2" \
    --config_path $config_path \
    --task "mcqa" \
    --num_frames 16 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path 
