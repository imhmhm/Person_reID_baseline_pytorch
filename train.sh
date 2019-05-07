python train.py \
--name ft_ResNet50_BT-no_bias_bn_cls-b16x4_adam_warmup_stride1_lsr_re \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--lsr \
--erasing_p 0.5 \
--warmup \
--adam;
# --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup;
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

# python train.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup_stride1 \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --warmup \
# --adam;
#
# python train.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup_lsr \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --lsr \
# --warmup \
# --adam;
#
# python train.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup_stride1_lsr \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --lsr \
# --warmup \
# --adam;
#
# python train.py \
# --name ft_ResNet50_BT_b16x4_adam_warmup_stride1_lsr_re \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 1 \
# --lsr \
# --erasing_p 0.5 \
# --warmup \
# --adam;
