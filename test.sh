python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 69;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 109;

python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 149;

python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 69;

python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 109;

python test.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_mixup_[60_100_140] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 149;
