python train.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--margin 1.2 \
--lr 0.00035 \
--adam;

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

python train.py \
--name ft_ResNet50_b16x4_adam_quadOnly_mg1.2_hd_lr2e-4 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--margin 1.2 \
--lr 0.0002 \
--adam;
