python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 69;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 199;

python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 69;

python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 199;
