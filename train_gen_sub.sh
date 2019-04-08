# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense


python train_gen_sub.py --gpu_ids 0 \
                        --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                        --gen_name gen_train_7p_v4_x1 \
                        --eps 0.4 \
                        --name ft_ResNet50_gen_e0.4_7p_v4_b128_spl_2+2_sub_tri_1:0.5:0.1 --train_all --batchsize 128 \
                        --num_per_id 4 \
                        --prop_real 2 \
                        --prop_gen 2 \
                        --sub 2048 \
                        --triplet \
                        --tri_weight 0.1 \
                        --sub_weight 0.5;

python train_gen_sub.py --gpu_ids 0 \
                        --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                        --gen_name gen_train_7p_v4_x1 \
                        --eps 0.4 \
                        --name ft_ResNet50_gen_e0.4_7p_v4_b128_spl_3+1_sub_tri_1:0.3:0.1 --train_all --batchsize 128 \
                        --num_per_id 4 \
                        --prop_real 3 \
                        --prop_gen 1 \
                        --sub 2048 \
                        --triplet \
                        --tri_weight 0.1 \
                        --sub_weight 0.3;

python train_gen_sub.py --gpu_ids 0 \
                        --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                        --gen_name gen_train_7p_v4_x1 \
                        --eps 0.4 \
                        --name ft_ResNet50_gen_e0.4_7p_v4_b128_spl_3+1_sub_tri_1:0.1:0.1 --train_all --batchsize 128 \
                        --num_per_id 4 \
                        --prop_real 3 \
                        --prop_gen 1 \
                        --sub 2048 \
                        --triplet \
                        --tri_weight 0.1 \
                        --sub_weight 0.1
