python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 59;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_stitch_test_lam0.5_pre \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;
