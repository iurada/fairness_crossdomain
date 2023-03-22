python main.py^
 --experiment .\experiments\landmark_detection\baseline\^
 --dataset .\datasets\landmark_detection\UTKFace\^
 --data_path .\data\utkface\^
 --max_iters 35000^
 --batch_size 32^
 --num_workers 1^
 --validate_every=500^
 --tracked_metrics mGS^
 --model_selection mGS