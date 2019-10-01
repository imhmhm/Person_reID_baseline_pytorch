# DukeMTMC-reID / Market / cuhk03-np / MSMT17_V2
python train.py \
--name ft_ResNet50_b16x4_SGD_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.01;
#--triplet;
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

# DukeMTMC-reID / Market / cuhk03-np / MSMT17_V2


# python train.py \
# --name ft_ResNet50_b16x4_tri1.0_SGD_msmt17 \
# --data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 120 \
# --lr 0.01 \
# --triplet;


python train.py \
--name ft_ResNet50_b16x4_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.00035 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_stride1_lsr_tri1.0_warmup_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.00035 \
--lsr \
--warmup \
--triplet \
--adam;

# python train.py \
# --name ft_ResNet50_b16x4_warmup_adam_msmt17 \
# --data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
# --train_all \
# --batchsize 64 \
# --droprate 0 \
# --use_sampler \
# --num_per_id 4 \
# --stride 2 \
# --epoch 120 \
# --lr 0.00035 \
# --warmup \
# --adam;

#================
# test
#================
bash test.sh
bash evaluate.sh

python train.py \
--name ft_ResNet50_b16x4_stride1_lsr_tri1.0_warmup_RE_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.00035 \
--lsr \
--warmup \
--triplet \
--erasing_p 0.5 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_stride1_SGD_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.01;

python train.py \
--name ft_ResNet50_b16x4_stride1_lsr_tri1.0_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.00035 \
--lsr \
--triplet \
--adam;

python train.py \
--name ft_ResNet50_b16x4_stride1_tri1.0_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 1 \
--epoch 120 \
--lr 0.00035 \
--triplet \
--adam;

python train.py \
--name ft_ResNet50_b16x4_tri1.0_adam_msmt17 \
--data_dir /home/hmhm/reid/MSMT17_V2/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--lr 0.00035 \
--triplet \
--adam;

bash test_bunch.sh
bash evaluate_bunch.sh
