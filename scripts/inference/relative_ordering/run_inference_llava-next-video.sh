#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/relative_ordering/llava-next-video-7b.json"

python inference.py \
    --model "llava-next-video" \
    --model_path "models/weights/LLaVA-NeXT-Video-7B-DPO" \
    --task "relative_ordering" \
    --num_frames 16 \
    --mm_spatial_pool_mode "average" \
    --mm_newline_position "no_token" \
    --mm_pooling_position "after"\
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path 
