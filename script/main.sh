CUDA_VISIBLE_DEVICES=1 python main.py   --option video \
                                        --path_video data/test/16.mp4 \
                                        --name_model LSTM \
                                        --replicate 30 \
                                        --num_train 0 \
                                        --num_ckp best_epoch \
                                        --save_video True \
                                        --path_save_video results/video

