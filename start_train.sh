export CUDA_VISIBLE_DEVICES="7"
cd /home/hongruixin/Siemens-project
python train.py --comment multitask_dropout_seed2023 \
--output_dir Outputs/ \
--train_path data/data_v1/ \
--test_path data/data_v1/real_image_wo_repeat/ \
--base_lr 0.001 \
--batch_size 96 \
--max_epoch 200 \
--val_epoch 1 \
--val_epoch_train 5 \
--save_epoch 15 \
--save_start_epoch 80 \
--img_size 360 \
--feature_len 2048 \
--dropout_rate 0.5 \
--seed 2023