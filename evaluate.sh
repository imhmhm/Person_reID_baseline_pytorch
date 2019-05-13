python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set Market \
--which_epoch 59;

# DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set Market \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri_insNorm_lr1e-4_[20_40_80] \
--test_set DukeMTMC-reID \
--which_epoch 119;
