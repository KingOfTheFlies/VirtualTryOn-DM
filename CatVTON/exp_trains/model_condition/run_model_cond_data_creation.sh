#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python model_cond_data_creation.py \
    --dataset vitonhd \
    --data_root_path /path/to/VTON-HD \
    --output_dir output_dir \
    --resume_path /path/to/checkpoint  \
    --sub_folder "test" \
    --device cuda \
    --load_pth_attn True \
    --guidance_scale 2.5 \
    --dataloader_num_workers 8 \
    --batch_size 64 \
    --seed 555 \
    --mixed_precision no \
    --allow_tf32 \
    --repaint \
    --eval_pair