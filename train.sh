python train.py \
--name ft_ResNet50_b32x4_tri0.1 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.01 \
--wt_tri 0.1 \
--triplet;
#--adam;

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
# DukeMTMC-reID / Market

python train.py \
--name ft_ResNet50_b32x4_tri1.0 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.01 \
--wt_tri 1.0 \
--triplet;
#--adam;


python train.py \
--name ft_ResNet50_b32x4_tri0.1_lsr \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.01 \
--lsr \
--wt_tri 0.1 \
--triplet;
#--adam;

python train.py \
--name ft_ResNet50_b32x4_tri0.1_stride1 \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.01 \
--wt_tri 0.1 \
--triplet;

python train.py \
--name ft_ResNet50_b32x4_tri0.1_warmup \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 128 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.01 \
--warmup \
--wt_tri 0.1 \
--triplet;
