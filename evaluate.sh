python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 119;


python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_set Market \
--which_epoch 119;


python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_set Market \
--which_epoch 119;


python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_set DukeMTMC-reID \
--which_epoch 119;
