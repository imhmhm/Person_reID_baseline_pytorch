# python test.py \
# --name ft_ResNet50_b16x4_adam_[40_80] \
# --test_dir /home/tianlab/hengheng/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 49;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 119;

#

# python test.py \
# --name ft_ResNet50_b16x4_adam_freezeL1L2 \
# --test_dir /home/tianlab/hengheng/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 49;

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;
