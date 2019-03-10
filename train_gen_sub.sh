# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

python train_gen_sub.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name gen_train_7p_v1_x1 \
                    --eps 0.4 \
                    --name ft_ResNet50_gen_e0.4_7p_v1_b128_spl_3+1_sub --train_all --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1;
                    #--mixup \
                    #--resume \
                    #--PCB \

python train_gen_sub.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name gen_train_7p_v1_x1 \
                    --eps 0.0 \
                    --name ft_ResNet50_gen_e0.0_7p_v1_b128_spl_3+1_sub --train_all --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1;
