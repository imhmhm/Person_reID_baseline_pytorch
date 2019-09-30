# python test.py \
# --name ft_ResNet50_b16x4_adam_[40_80] \
# --test_dir /home/tianlab/hengheng/reid \
# --test_set DukeMTMC-reID \
# --which_epoch 49;

# --multi \
# --PCB \
# DukeMTMC-reID / Market / cuhk03-np
#--which_epoch \

#===============================================
python test.py \
--name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 119;

#===============================================

python test.py \
--name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 119;

#===============================================

python test.py \
--name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 59;

python test.py \
--name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 99;

python test.py \
--name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
--test_dir /home/tianlab/hengheng/reid \
--test_set cuhk03-np \
--which_epoch 119;
