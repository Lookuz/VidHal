#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/naive_ordering/videollama2-7b.json"

python inference.py \
    --model "videollama2" \
    --model_path "models/weights/VideoLLaMA2-7B" \
    --task "naive_ordering" \
    --num_frames 8 \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
