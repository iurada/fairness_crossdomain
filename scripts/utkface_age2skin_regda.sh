#!/bin/bash

python main.py \
--experiment experiments/landmark_detection/RegDA \
--dataset datasets/landmark_detection/UTKFace \
--experiment age2skin \
--data_path data/utkface \
--regda_pretrain_iters 35000 \
--max_iters 15000 \
--batch_size 32 \
--num_workers 4 \
--validate_every 500 \
--tracked_metrics SDR MGS mGS DS DeltaDTO HF \
--model_selection mGS \
--baseline_MGS 86.67 \
--baseline_DS 8.70 \
--baseline_DTO 22.11 \
--regda_margin 4.0 \
--regda_tradeoff 0.5