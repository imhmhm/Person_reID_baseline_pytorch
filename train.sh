python train.py \
--name ft_ResNet50_b16x4_adam_vert_duke \
--data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.00035 \
--mixup \
--adam;

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

# python train.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_re_duke \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 120 \
# --lr 0.00035 \
# --triplet \
# --lsr \
# --erasing_p 0.5 \
# --adam;
#
#
# python train.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_lsr0.1_stride1_re_duke \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --epoch 120 \
# --lr 0.00035 \
# --triplet \
# --lsr \
# --erasing_p 0.5 \
# --adam;
#
# python train.py \
# --name ft_ResNet50_b16x4_adam_xent+tri_warmup_lsr0.1_stride1_duke \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --epoch 120 \
# --lr 0.00035 \
# --triplet \
# --warmup \
# --lsr \
# --adam;
#
