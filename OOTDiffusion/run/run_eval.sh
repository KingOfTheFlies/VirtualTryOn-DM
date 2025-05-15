#!/bin/bash

python ootd_hd_eval.py \
                    --dataset_dir 'dataset_dir' \
                    --load_height 1024 \
                    --load_width 768 \
                    --dataset_list 'test_pairs.txt' \
                    --dataset_mode 'test' \
                    --batch_size 16 \
                    --train_batch_size 16 \
                    --do_classifier_free_guidance True \
                    --image_guidance_scale 1.5 \
                    --save_outputs True \
                    --save_path "save_path"