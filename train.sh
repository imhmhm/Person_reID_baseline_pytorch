python train.py \
--name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_beta \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lsr \
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

# python train.py \
# --name ft_ResNet50_BT_b16x4_adam_tri_warmup_insNorm_lr2e-4_ep200_[70_120] \
# --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --triplet \
# --num_per_id 4 \
# --stride 2 \
# --epoch 200 \
# --warmup \
# --adam;
