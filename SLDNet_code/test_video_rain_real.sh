CUDA_VISIBLE_DEVICES=1 python test_video_rain_real.py -method P401_video_rain_self -epoch 95 -dataset Video_rain -task RainRemoval/original -data_dir ../../Formal_code/dataset/Rain_video_synthesis_v0822_train1800/ -model_name derain_self_v1  -checkpoint_dir ./checkpoints/P401_video_rain_self/ -list_filename ./lists/video_rain_removal_test_real.txt -iter 1

