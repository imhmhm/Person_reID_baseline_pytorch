python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set Market \
--which_epoch 59;

# DukeMTMC-reID / Market

python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set Market \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set Market \
--which_epoch 119;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set DukeMTMC-reID \
--which_epoch 59;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate.py \
--name ft_ResNet50_BT_b16x4_sgd0.01_[30_70]_tri \
--test_set DukeMTMC-reID \
--which_epoch 119;
