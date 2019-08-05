# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

# gen_train_7p_v4_x1 \

python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
                    --gen_name duke_train_7p_v4_v2_x1 \
                    --eps 0.4 \
                    --name duke_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri_fp16_v2 --train_all --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --fp16 \
                    --triplet;
                    # --erasing_p 0.5
                    # --use_dense \

                    # --PCB \
                    # --use_resnext
                    # --adam
                    #--mixup \
                    #--resume \
                    #--PCB \
python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
                    --gen_name duke_train_7p_v4_v2_x1 \
                    --eps 0.2 \
                    --name duke_ResNet50_gen_e0.2_branch_7p_v4_b128_spl_3+1_tri_fp16_v2 --train_all --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --fp16 \
                    --triplet;
