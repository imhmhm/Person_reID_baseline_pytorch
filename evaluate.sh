# python evaluate.py \
# --name ft_ResNet50_baseline_adam \
# --test_set Market \
# --which_epoch 59;
#
# # DukeMTMC-reID / Market
#
# python evaluate.py \
# --name ft_ResNet50_baseline_adam \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate.py \
# --name ft_ResNet50_noise0.1_adam \
# --test_set Market \
# --which_epoch 59;
#
# python evaluate.py \
# --name ft_ResNet50_noise0.1_adam \
# --test_set Market \
# --which_epoch 99;

python evaluate.py \
--name ft_ResNet50_noise0.3  \
--test_set Market \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_noise0.3 \
--test_set Market \
--which_epoch 99;
#
# python evaluate.py \
# --name ft_ResNet50_noise0.5_adam \
# --test_set Market \
# --which_epoch 59;
#
# python evaluate.py \
# --name ft_ResNet50_noise0.5_adam \
# --test_set Market \
# --which_epoch 99;
