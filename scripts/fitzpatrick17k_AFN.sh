#!/bin/bash

python main.py \
--experiment experiments/classification/AFN \
--dataset datasets/classification/Fitzpatrick17k \
--data_path data/fitzpatrick17k \
--max_iters 46875 \
--batch_size 64 \
--num_workers 4 \
--validate_every 500 \
--tracked_metrics Acc MGA mGA DA DEO DEOdds DeltaDTO HF \
--model_selection mGA \
--baseline_MGA 94.45 \
--baseline_DA 7.19 \
--baseline_DTO 12.83 \
--afn_type SAFN