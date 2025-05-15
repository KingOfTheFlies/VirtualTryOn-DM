#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference_w_cloth_mask.py \
    --dataset dresscode \
    --data_root_path data_root_path \
    --output_dir output_dir \
    --resume_path pth/to/resume_path.pth \
    --device cuda \
    --load_pth_attn True \
    --guidance_scale 1 \
    --dataloader_num_workers 8 \
    --batch_size 64 \
    --seed 555 \
    --mixed_precision no \
    --allow_tf32 \
    --eval_pair \
    --repaint 