python evaluate.py \
--name ft_ResNet50_GP_b32x4_sgd0.005_all_relu \
--test_set Market \
--which_epoch 59;

# DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_GP_b32x4_sgd0.005_all_relu \
--test_set Market \
--which_epoch 99;

# python evaluate.py \
# --name ft_ResNet50_GP_b32x4_sgd0.01_relu \
# --test_set Market \
# --which_epoch 139;

python evaluate.py \
--name ft_ResNet50_GP_b32x4_sgd0.005_all_relu \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_GP_b32x4_sgd0.005_all_relu \
--test_set DukeMTMC-reID \
--which_epoch 99;

# python evaluate.py \
# --name ft_ResNet50_GP_b32x4_sgd0.01_relu \
# --test_set DukeMTMC-reID \
# --which_epoch 139;
