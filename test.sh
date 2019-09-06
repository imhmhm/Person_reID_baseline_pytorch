python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

# --multi \
# --PCB \
# DukeMTMC-reID / Market
#--which_epoch \

python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;


python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_vert_metricOnly_easy50hard_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_duke \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--which_epoch 119;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_adam_metricOnly_bn_market \
--test_dir /home/hmhm/reid \
--test_set Market \
--which_epoch 119;

bash evaluate.sh;
