# python test.py \
# --name ft_ResNet50_baseline_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch \
# --test_set Market \
# --which_epoch 59;
# #--name ft_ResNet50_1toM_s0.1_p0.5 \
# #--data_dir /home/hmhm/reid/Market/pytorch_1toM \
# #--which_epoch \
#
# python test.py \
# --name ft_ResNet50_baseline_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch \
# --test_set Market \
# --which_epoch 99;
#
# python test.py \
# --name ft_ResNet50_noise0.1_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.1 \
# --test_set Market \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_noise0.1_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.1 \
# --test_set Market \
# --which_epoch 99;

python test.py \
--name ft_ResNet50_noise0.3 \
--test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.3 \
--test_set Market \
--which_epoch 59;

python test.py \
--name ft_ResNet50_noise0.3 \
--test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.3 \
--test_set Market \
--which_epoch 99;

# python test.py \
# --name ft_ResNet50_noise0.5_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.5 \
# --test_set Market \
# --which_epoch 59;
#
# python test.py \
# --name ft_ResNet50_noise0.5_adam \
# --test_dir /home/tianlab/hengheng/reid/Market/pytorch_noise_0.5 \
# --test_set Market \
# --which_epoch 99;
