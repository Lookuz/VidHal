#!/bin/sh

annotations_path="vidhal/annotations.json"
videos_path="vidhal/videos"
options_path="vidhal/options.json"
predictions_path="outputs/inference/naive_ordering/random.json"
save_path="outputs/evaluation/naive_ordering/random.json"

python evaluate.py \
    --task "naive_ordering" \
    --annotations_path $annotations_path \
    --videos_path $videos_path \
    --predictions_path $predictions_path \
    --save_path $save_path \
    --options_path $options_path
