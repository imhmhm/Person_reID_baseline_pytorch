python train.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_re_triOnly \
--data_dir /home/hmhm/reid/Market/pytorch \
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
# --lr 0.01;
# # --triplet \
# # --erasing_p 0.5 \
# # --mixup;
# # --num_per_id 4 \
# # --PCB \
# # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

python train.py \
--name ft_ResNet50_b16x4_adam_warmup_lsr_triOnly \
--data_dir /home/hmhm/reid/Market/pytorch \
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
--name ft_ResNet50_b16x4_adam_warmup_triOnly \
--data_dir /home/hmhm/reid/Market/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--triplet \
--num_per_id 4 \
--stride 2 \
--warmup \
--adam;
