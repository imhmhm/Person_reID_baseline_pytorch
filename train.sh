python train.py \
--name ft_ResNet50_b8x4_stride1_SGD_cuhk03 \
--data_dir /home/hmhm/reid/cuhk03-np/pytorch \
--train_all \
--batchsize 32 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.01;
#--triplet;
#--adam;

# # --margin 1.2 \
# # --lsr \
# # --triplet \
# # --warmup \
# # --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup;
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

# DukeMTMC-reID / Market / cuhk03-np


# python train.py \
# --name ft_ResNet50_b16x4_tri0.15_stride1_SGD_duke \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --epoch 120 \
# --lr 0.01 \
# --wt_tri 0.15 \
# --triplet;
#


#================
# test
#================
bash test.sh
bash evaluate.sh
