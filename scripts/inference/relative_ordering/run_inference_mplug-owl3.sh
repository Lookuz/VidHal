#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/relative_ordering/mplug-owl3-7b.json"

python inference.py \
    --model "mplug_owl3" \
    --model_path "models/weights/mPLUG-Owl3-7B-240728" \
    --task "relative_ordering" \
    --num_frames 16 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path 
