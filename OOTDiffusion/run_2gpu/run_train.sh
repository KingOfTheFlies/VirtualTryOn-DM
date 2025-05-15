#!/bin/bash

# bash run_train.sh > train_with_acc_bs64_out.txt 2>&1, nohup ./run_train.sh > train_with_acc_bs64.out 2>&1 &
accelerate launch run_mytrain.py \
                                --load_height 512 \
                                --load_width 384 \
                                --dataset_list 'train_pairs.txt' \
                                --dataset_mode 'train' \
                                --batch_size 16 \
                                --train_batch_size 16 \
                                --num_train_epochs 500 \
                                --gradient_accumulation_steps 2 \
                                # --mixed_precision "fp16" \
