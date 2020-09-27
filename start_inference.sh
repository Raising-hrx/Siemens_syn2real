export CUDA_VISIBLE_DEVICES="0"
cd /home/hongruixin/Siemens-project
python inference.py --output_dir Outputs/multitask_dropout_seed2021-2020-09-27_14-06 \
--model model_dict_best.pth \
--batch_size 16 \
--img_size 360 \
--feature_len 2048 \
--dropout_rate 0.5 \
--run_on_dataset \
--test_path data/data_v1/real_image_wo_repeat/ \
# --test_path data/data_v0/Raw/real_images/F58001104949503200002/img_3461.png \
