# python train.py \
# --name ft_ResNet50_baseline \
# --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
# --train_all \
# --droprate 0.0 \
# --lr 0.01;


# python train.py \
# --name ft_ResNet50_noise0.1 \
# --data_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.1 \
# --train_all \
# --droprate 0.0 \
# --lr 0.01;

# python train.py \
# --name ft_ResNet50_noise0.3 \
# --data_dir /home/tianlab/hengheng/reid/Market/pytorch_noise \
# --train_all \
# --droprate 0.0 \
# --lr 0.01;
#
python train.py \
--name ft_ResNet50_noise0.5 \
--data_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.5 \
--train_all \
--droprate 0.0 \
--lr 0.01 \
