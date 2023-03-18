#!/bin/bash

python main.py \
--experiment experiments/classification/baseline \
--dataset datasets/classification/CelebA \
--data_path data/celeba \
--max_iters 3000 \
--batch_size 64 \
--num_workers 1 \
--validate_every 500 \
--tracked_metrics Acc MGA mGA DA DEO DEOdds DTO HF \
--model_selection mGA \
--baseline_MGA 0 \
--baseline_DA 100 \
--target_attribute 3 \
--protected_attribute 20