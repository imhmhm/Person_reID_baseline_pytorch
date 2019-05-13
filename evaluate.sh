python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set Market \
--which_epoch 69;

# # DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set Market \
--which_epoch 109;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set Market \
--which_epoch 149;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set DukeMTMC-reID \
--which_epoch 69;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set DukeMTMC-reID \
--which_epoch 109;

python evaluate.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_set DukeMTMC-reID \
--which_epoch 149;
