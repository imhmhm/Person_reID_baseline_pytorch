python train.py \
       --name ft_ResNet50_b128_spl_32x4_re0.5 \
       --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
       --train_all \
       --batchsize 128 \
       --num_per_id 4 \
       --erasing_p 0.5 \
       # --num_per_id 8;
       # --PCB \
       #--mixup \
       #--name ft_ResNet50_128_relu \
       #--data_dir /home/tianlab/hengheng/reid/Market/pytorch \
       #--train_all \

# python train.py \
#      --name ft_ResNet50_b128_spl_8x16 \
#      --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#      --train_all \
#      --batchsize 128 \
#      --num_per_id 16;
