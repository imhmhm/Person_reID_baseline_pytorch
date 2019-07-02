python train.py \
--name ft_ResNet50_b16x4_adam_warmup_freezeL1L2 \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--warmup \
--lr 0.00035 \
--adam;
# --fp16 \

# --triplet \
# --resume ft_ResNet50_BT_b16x4_adam_warmup_stride1_lsr_tri \
# --warmup \
# --lsr \
# # --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup \
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

python train.py \
--name ft_ResNet50_b16x4_adam_[40_80]_lr2.5e-4_freezeL1L2 \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.00025 \
--adam;
