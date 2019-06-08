python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 59;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_test_v2_margin_all_0.3 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;
