python train.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_lsr_re_tri \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--lsr \
--erasing_p 0.5 \
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

python train.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_lsr_tri \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--lsr \
--warmup \
--adam;

python train.py \
--name ft_ResNet50_BT_b16x4_adam_warmup_tri \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--warmup \
--adam;
