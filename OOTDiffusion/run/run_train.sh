#!/bin/bash

# bash run_train.sh > raw_train.txt 2>&1, nohup run_train.sh > train_with_base_init.out 2>&1 &
accelerate launch run_mytrain.py \
                                --load_height 512 \
                                --load_width 384 \
                                --dataset_list 'train_pairs.txt' \
                                --dataset_mode 'train' \
                                --batch_size 16 \
                                --train_batch_size 16 \
                                --num_train_epochs 200 \
                                # --mixed_precision "fp16" \
