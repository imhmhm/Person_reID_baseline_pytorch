python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 69;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 199;

python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 69;

python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 199;
