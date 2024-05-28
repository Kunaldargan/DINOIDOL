#CUDA_VISIBLE_DEVICES=7 python3 projects/IDOL/train_net_video_EV18.py --config-file  projects/IDOL/configs/EV18_r50.yaml --num-gpus 1 #--eval-only
CUDA_VISIBLE_DEVICES=6 python3 train_net_video_EV18_DINOIDOL.py --config-file projects/IDOL/configs/EV18_DINOIDOL_r50.yaml --num-gpus 1 #--eval-only
# -m pdb -c continue
