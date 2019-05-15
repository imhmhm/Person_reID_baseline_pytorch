python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 69;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--name ft_ResNet50_1toM_s0.1_p0.5 \
#--data_dir /home/hmhm/reid/Market/pytorch_1toM \
#--which_epoch \

python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set Market \
--which_epoch 199;

python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 69;

python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;

python test.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
--test_dir /home/tianlab/hengheng/reid \
--test_set DukeMTMC-reID \
--which_epoch 199;
