# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

# 1501_train_7p_v4_x1 \ duke_train_7p_v4_5views_v2_x1

# DukeMTMC-reID / Market / cuhk03-np

python train_gen.py \
--gpu_ids 0 \
--data_dir /home/hmhm/reid/cuhk03-np/pytorch \
--gen_name cuhk03_train_7p_v4_5views_x1 \
--lsr \
--eps_gen 0.4 \
--eps_real 0.0 \
--name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
--train_all \
--batchsize 64 \
--num_per_id 4 \
--prop_real 3 \
--prop_gen 1 \
--stride 1 \
--wt_tri 0.3 \
--lr 0.01 \
--triplet;

# # --wt_tri 0.01 \
# # --triplet;
# # --adam \
# # --fp16 \
# # --erasing_p 0.5 \
# # --use_dense \
# # --PCB \
# # --use_resnext \
# #--mixup \
# #--resume \


python train_gen.py \
--gpu_ids 0 \
--data_dir /home/hmhm/reid/cuhk03-np/pytorch \
--gen_name cuhk03_train_7p_v4_5views_x1 \
--lsr \
--eps_gen 0.4 \
--eps_real 0.0 \
--name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
--train_all \
--batchsize 64 \
--num_per_id 4 \
--prop_real 3 \
--prop_gen 1 \
--stride 1 \
--wt_tri 0.1 \
--lr 0.01 \
--triplet;

# --wt_tri 0.2 \
# --triplet;


python train_gen.py \
--gpu_ids 0 \
--data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
--gen_name duke_train_7p_v4_5views_v2_x1 \
--lsr \
--eps_gen 0.4 \
--eps_real 0.0 \
--name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
--train_all \
--batchsize 64 \
--num_per_id 4 \
--prop_real 3 \
--prop_gen 1 \
--stride 1 \
--wt_tri 0.1 \
--lr 0.01 \
--triplet;

python train_gen.py \
--gpu_ids 0 \
--data_dir /home/hmhm/reid/Market/pytorch \
--gen_name 1501_train_7p_v4_x1 \
--lsr \
--eps_gen 0.4 \
--eps_real 0.0 \
--name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
--train_all \
--batchsize 64 \
--num_per_id 4 \
--prop_real 3 \
--prop_gen 1 \
--stride 1 \
--wt_tri 0.1 \
--lr 0.01 \
--triplet;


#=============
# test
#=============
bash test_gen.sh;
#
bash evaluate_gen.sh;
#
# bash train_gen_v1.sh;

#=============
#=============
