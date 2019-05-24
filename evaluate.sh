python evaluate.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_set Market \
--which_epoch 69;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_set Market \
--which_epoch 119;

# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_mixup_metric_test \
# --test_set Market \
# --which_epoch 199;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_set DukeMTMC-reID \
--which_epoch 69;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_set DukeMTMC-reID \
--which_epoch 119;

# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_mixup_metric_test \
# --test_set DukeMTMC-reID \
# --which_epoch 199;
