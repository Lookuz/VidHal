#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
predictions_path="outputs/inference/mcqa/random.json"
save_path="outputs/evaluation/mcqa/random.json"

python evaluate.py \
    --task "mcqa" \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --predictions_path $predictions_path \
    --save_path $save_path \
    --options_path $options_path
