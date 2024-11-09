#!/bin/bash

python main.py \
--experiment experiments/classification/baseline \
--dataset datasets/classification/CelebA \
--data_path /data/datasets/celeba \
--max_iters 46875 \
--batch_size 64 \
--num_workers 4 \
--validate_every 500 \
--tracked_metrics Acc MGA mGA DA DEO DEOdds DTO \
--model_selection mGA \
--target_attribute 3 \
--protected_attribute 20 \
--image_size 128