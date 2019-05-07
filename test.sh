# python test.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --which_epoch 59;
#
# # --multi \
# # --PCB \
# # DukeMTMC-reID / Market
# #--name ft_ResNet50_1toM_s0.1_p0.5 \
# #--data_dir /home/hmhm/reid/Market/pytorch_1toM \
# #--which_epoch \
#
# python test.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --which_epoch 99;

python test.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_stride1_lsr_re \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

# python test.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 99;

# python test.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup_stride1_lsr_re \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
