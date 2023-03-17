python main.py^
 --experiment .\experiments\landmark_detection\baseline\^
 --dataset .\datasets\landmark_detection\UTKFace\^
 --data_path data^
 --max_iters 46875^
 --batch_size 64^
 --num_workers 1^
 --validate_every=500^
 --tracked_metrics HF^
 --model_selection HF^
 --baseline_MGS 0^
 --baseline_DS 100^
 --SDR_threshold 0.08^
 --p1_NME_index 36^
 --p2_NME_index 45