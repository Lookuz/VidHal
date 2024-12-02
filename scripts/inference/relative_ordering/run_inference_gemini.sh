#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/relative_ordering/gemini.json"

python inference.py \
    --model "gemini-1.5-flash" \
    --api_key "<your_api_key>" \
    --task "relative_ordering" \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
