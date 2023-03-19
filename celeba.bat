python main.py^
 --experiment .\experiments\classification\GroupDRO\^
 --dataset .\datasets\classification\CelebA\^
 --data_path .\data\celeba\^
 --max_iters 3000^
 --batch_size 64^
 --num_workers 1^
 --validate_every=500^
 --tracked_metrics mGA HF^
 --model_selection mGA^
 --baseline_MGA 0^
 --baseline_DA 100^
 --target_attribute 3^
 --protected_attribute 20