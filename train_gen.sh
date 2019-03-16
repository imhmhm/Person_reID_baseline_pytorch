# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_mixup_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_mixup_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
#                     # --triplet \
#                     # --adam
#                     #--mixup \
#                     #--resume \
#                     #--PCB \
#
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_mixup_noreid_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_mixup_noreid_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;

# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_mixup_40epoch_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_mixup_40epoch_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
#
python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name gen_train_7p_v4_x1 \
                    --eps 0.4 \
                    --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_1+3 --train_all --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 1 \
                    --prop_gen 3;

# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_view0 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
