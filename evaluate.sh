# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_freezeL1L2 \
# --test_set DukeMTMC-reID \
# --which_epoch 49;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set Market \
--which_epoch 119;

# python evaluate.py \
# --name ft_ResNet50_b16x4_adam_freezeL1L2 \
# --test_set DukeMTMC-reID \
# --which_epoch 49;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_set DukeMTMC-reID \
--which_epoch 119;
