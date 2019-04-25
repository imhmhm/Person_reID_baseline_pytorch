python train.py \
--name ft_ResNet50_BT_b16x4_adam_tri \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--stride 2 \
--adam;
# --mixup;
# --num_per_id 4 \
# --erasing_p 0.5 \
# --num_per_id 8;
# --PCB \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

python train.py \
--name ft_ResNet50_BT_b16x4_adam_tri_stride1 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--stride 1 \
--adam;

python train.py \
--name ft_ResNet50_BT_b16x4_adam_tri_lsr \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--stride 2 \
--lsr \
--adam;
