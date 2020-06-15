#CUDA_VISIBLE_DEVICES=1 python train_video_rain.py -datasets_tasks W3_D1_C1_I1_sh

CUDA_VISIBLE_DEVICES=6 python -u train_video_rain.py -checkpoint_dir ./checkpoints/P202_video_rain_pretrain_formal_code/ -data_dir Video_rain -list_filename ./lists/video_rain_removal_train.txt -crop_size 64 
