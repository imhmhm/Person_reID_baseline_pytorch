python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set Market \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_set DukeMTMC-reID \
--which_epoch 119;
