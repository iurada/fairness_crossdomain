python main.py^
 --experiment .\experiments\classification\baseline\^
 --dataset .\datasets\classification\CelebA\^
 --data_path data^
 --max_iters 46875^
 --batch_size 64^
 --num_workers 1^
 --validate_every=500^
 --tracked_metrics HF^
 --model_selection HF^
 --baseline_MGA 0^
 --baseline_DA 100^
 --transfer_experiment age2gender