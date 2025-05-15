#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
       --dataset_dir "dataset_dir" \
       --base_ckpt "runwayml/stable-diffusion-inpainting" \
       --attn_ckpt "path/to/attn_ckpt" \
       --opt_ckpt ".path/to/opt_ckpt" \
       --save_chkp_every 6400 \
       --save_attn_ckpt_path save_attn_ckpt_path \
       --device "cuda" \
       --compile True \
       --batch_size 32 \
       --max_train_steps 64500 \
       --learning_rate 0.0005 \
       --cfg_dropout_prob 0.1 \
       --log_step 256 \
       --wandb_project "wandb_project" \
       --wandb_run_name "wandb_run_name" \
       --wandb_api_key "API"