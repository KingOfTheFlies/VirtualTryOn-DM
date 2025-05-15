#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --dataset dresscode \
    --data_root_path data_root_path \
    --output_dir output_dir \
    --resume_path zhengchong/CatVTON \
    --load_pth_attn True \
    --device cuda \
    --guidance_scale 2.5 \
    --dataloader_num_workers 8 \
    --batch_size 1 \
    --seed 555 \
    --mixed_precision no \
    --allow_tf32 \
    --concat_eval_results \
    --repaint \
    --eval_pair