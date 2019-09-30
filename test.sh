# #===============================================
# python test.py \
# --name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 99;

python test.py \
--name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
--test_dir /home/hmhm/reid \
--test_set cuhk03-np \
--which_epoch 119;

# #===============================================
#
# python test.py \
# --name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_b8x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 119;
#
# #===============================================
#
# python test.py \
# --name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_b16x4_tri0.15_stride1_SGD_cuhk03 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --which_epoch 119;

#========================
#========================
bash evaluate.sh;
