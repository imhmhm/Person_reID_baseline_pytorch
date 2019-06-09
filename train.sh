python train.py \
--name ft_ResNet50_b16x4_adam_mixup_test_beta3.0_ep200_80_150 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 200 \
--mixup \
--adam;

# --triplet \
# # --warmup;
# # --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup;
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \
