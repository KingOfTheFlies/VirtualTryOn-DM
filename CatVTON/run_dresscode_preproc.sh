#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python preprocess_agnostic_mask.py \
    --data_root_path "data_root_path/DressCode"