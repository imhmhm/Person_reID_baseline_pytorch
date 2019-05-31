python train_multi.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_gpNorm_L1_B3 \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--warmup \
--adam;

# --stride 1 \
# --lsr \
# --erasing_p 0.5 \
# --warmup \
# --adam;
# --lr 0.01;
# --PCB \
# --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
