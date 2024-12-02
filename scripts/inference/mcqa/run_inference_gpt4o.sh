#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
save_path="outputs/inference/mcqa/gpt4o.json"

python inference.py \
    --model "gpt-4o" \
    --api_key "<your_api_key>" \
    --task "mcqa" \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --save_path $save_path \
    --options_path $options_path
