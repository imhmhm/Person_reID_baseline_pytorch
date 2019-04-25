python evaluate.py \
--name ft_ResNet50_b32x4_adam_tri_afterbn \
--test_set Market \
--which_epoch 59;

# DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b32x4_adam_tri_afterbn \
--test_set Market \
--which_epoch 99;

# python evaluate.py \
# --name ft_ResNet50_b32_ori \
# --test_set Market \
# --which_epoch 139;

python evaluate.py \
--name ft_ResNet50_b32x4_adam_tri_afterbn \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_b32x4_adam_tri_afterbn \
--test_set DukeMTMC-reID \
--which_epoch 99;

# python evaluate.py \
# --name ft_ResNet50_b32_ori \
# --test_set DukeMTMC-reID \
# --which_epoch 139;
