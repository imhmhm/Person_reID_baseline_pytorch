python train.py \
      --name ft_ResNet50_GP_b32_adam_step40 \
      --data_dir /home/hmhm/reid/Market/pytorch \
      --train_all \
      --batchsize 32 \
      --adam \
      --droprate 0;
       # --mixup \
       # --num_per_id 4 \
       # --erasing_p 0.5 \
       # --num_per_id 8;
       # --PCB \
       # --data_dir /home/hmhm/reid/DukeMTMC-reID/pytorch \

python train.py \
       --name ft_ResNet50_GP_b32_sgd_step40 \
       --data_dir /home/hmhm/reid/Market/pytorch \
       --train_all \
       --batchsize 32 \
       --droprate 0;
