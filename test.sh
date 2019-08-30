python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;


python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_stride1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_stride1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_stride1_re_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;

bash evaluate.sh;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_lsr0.1_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_lsr0.1_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_lsr0.1_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_re_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_re_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_re_stride1_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_warmup_stride1_re_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_warmup_stride1_re_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_warmup_stride1_re_duke \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
