python train.py \
--name ft_ResNet50_b16x4_adam_mixup_metric_test_v2 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--adam;

# --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup;
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

# python train.py \
# --name ft_ResNet50_b16x4_adam_mixup_lsr0.1_ep200_lr2e-4_[70_120] \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 200 \
# --adam \
# --lsr \
# --mixup;
#
# python train.py \
# --name ft_ResNet50_b16x4_adam_mixup_ep200_lr2e-4_[70_120] \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 200 \
# --adam \
# --mixup;
#
# python train.py \
# --name ft_ResNet50_b32x4_adam_mixup_ep200_lr2e-4_[70_120] \
# --data_dir /home/hmhm/reid/Market/pytorch \
# --train_all \
# --batchsize 128 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 200 \
# --adam \
# --mixup;
