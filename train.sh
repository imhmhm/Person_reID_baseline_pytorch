python train.py \
--name ft_ResNet50_GP_b32_adam_tri \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 32 \
--droprate 0 \
--triplet \
--adam;
# --mixup;
# --num_per_id 4 \
# --erasing_p 0.5 \
# --num_per_id 8;
# --PCB \
# --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

python train.py \
--name ft_ResNet50_GP_b16x4_adam_tri \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--adam;

python train.py \
--name ft_ResNet50_GP_b32x4_adam_tri \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--adam;

python train.py \
--name ft_ResNet50_GP_b8x4_adam_tri \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 32 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--triplet \
--adam;
