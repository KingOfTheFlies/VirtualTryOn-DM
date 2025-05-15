#!/bin/bash

python ootd_hd_eval.py \
                    --dataset_dir dataset_dir \
                    --load_height 512 \
                    --load_width 384 \
                    --dataset_list 'test_pairs.txt' \
                    --dataset_mode 'test' \
                    --batch_size 16 \
                    --train_batch_size 16 \
                    --do_classifier_free_guidance False \
                    --image_guidance_scale 0 \
                    --save_outputs True \
                    --save_path "save_path" \
                    --weights_folder_name "weights_folder_name"