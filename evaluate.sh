python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd \
--test_set Market \
--which_epoch 99;


python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd \
--test_set Market \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd_lr2e-4 \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd_lr2e-4 \
--test_set Market \
--which_epoch 99;


python evaluate.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd_lr2e-4 \
--test_set Market \
--which_epoch 119;

# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_stitch_metricOnly_50_[40_80] \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_stitch_metricOnly_50_[40_80] \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_stitch_metricOnly_50_[40_80] \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
