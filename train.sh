python train.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_duke \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 1.0 \
--wt_tri 1.0 \
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
--name ft_ResNet50_b16x4_adam_stitch_metric_duke_lr3e-4 \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 1.0 \
--wt_tri 1.0 \
--lr 0.0003 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_duke_lr2.5e-4 \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 1.0 \
--wt_tri 1.0 \
--lr 0.00025 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_wt0.8_duke \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 0.8 \
--wt_tri 1.0 \
--lr 0.00035 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_wt0.7_duke \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 0.7 \
--wt_tri 1.0 \
--lr 0.00035 \
--adam;

python train.py \
--name ft_ResNet50_b16x4_adam_stitch_metric_wt0.6_duke \
--data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
--train_all \
--batchsize 64 \
--droprate 0 \
--use_sampler \
--num_per_id 4 \
--stride 2 \
--epoch 120 \
--triplet \
--mixup \
--wt_xent 0.6 \
--wt_tri 1.0 \
--lr 0.00035 \
--adam;
