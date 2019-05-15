python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set Market \
--which_epoch 69;

# DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set Market \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set Market \
--which_epoch 199;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set DukeMTMC-reID \
--which_epoch 69;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set DukeMTMC-reID \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_set DukeMTMC-reID \
--which_epoch 199;
