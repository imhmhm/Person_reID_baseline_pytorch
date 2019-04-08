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
                    
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
#                     --gen_name duke_train_7p_v4_v2_x1 \
#                     --eps 0.4 \
#                     --name duke_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri_fp16_v2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 3 \
#                     --prop_gen 1 \
#                     --fp16 \
#                     --triplet;
#
# # gen_train_7p_v4_nomid_x1
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_nomid_x1 \
#                     --eps 0.4 \
#                     --name ft_res50_gen_e0.4_reid_branch_7p_v4_nomid_b128_spl_3+1_tri_fp16 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 3 \
#                     --prop_gen 1 \
#                     --fp16 \
#                     --triplet \
                    # --use_resnext \

                    #--use_dense \


# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/cuhk03_detected_np/pytorch \
#                     --gen_name cuhk03_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name cuhk03_res_gen_e0.4_reid_branch_7p_v4_b128_spl_3+1 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 3 \
#                     --prop_gen 1 \
#                     --lr 0.05;
#                     #--fp16 \
#                     #--triplet \
#                     #--use_dense \
#                     # --erasing_p 0.5
#
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/cuhk03_detected_np/pytorch \
#                     --gen_name cuhk03_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name cuhk03_res_gen_e0.4_reid_branch_7p_v4_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2 \
#                     --lr 0.05;
#
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/cuhk03_detected_np/pytorch \
#                     --gen_name cuhk03_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name cuhk03_res_gen_e0.4_reid_branch_7p_v4_b32_spl_2+2 --train_all --batchsize 32 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2 \
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_3+1_tri_1:0.1 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 3 \
#                     --prop_gen 1 \
#                     --triplet;

# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_2+1 --train_all --batchsize 128 \
#                     --num_per_id 3 \
#                     --prop_real 2 \
#                     --prop_gen 1;
#
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_head_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_head_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
#
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_mixup_40epoch_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_mixup_40epoch_b128_spl_2+2 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
# #
# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_x1 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_3+1_mul --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 3 \
#                     --prop_gen 1 \
#                     --triplet;

# python train_gen.py --gpu_ids 0 \
#                     --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
#                     --gen_name gen_train_7p_v4_view0 \
#                     --eps 0.4 \
#                     --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_2+2_view0 --train_all --batchsize 128 \
#                     --num_per_id 4 \
#                     --prop_real 2 \
#                     --prop_gen 2;
