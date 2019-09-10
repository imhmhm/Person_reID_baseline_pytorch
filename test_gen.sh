python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

                   # --use_dense \
                   # --PCB \
                   # --multi \
                   # --gen_query gen_query_7p_v4_3v \

                   # DukeMTMC-reID / Market / cuhk03_detected_np

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set Market \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

# #=============================================================================
# python test_gen.py --gpu_ids 0 \
#                    --test_dir /home/tianlab/hengheng/reid \
#                    --test_set Market \
#                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_SGD \
#                    --which_epoch 59 \
#                    --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
#                    --test_dir /home/tianlab/hengheng/reid \
#                    --test_set Market \
#                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_SGD \
#                    --which_epoch 99 \
#                    --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
#                    --test_dir /home/tianlab/hengheng/reid \
#                    --test_set Market \
#                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_SGD \
#                    --which_epoch 119 \
#                    --batchsize 32;

#====================================================================================================
#====================================================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

                   # --use_dense \
                   # --PCB \
                   # --multi \
                   # --gen_query gen_query_7p_v4_3v \

                   # DukeMTMC-reID / Market / cuhk03_detected_np

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.3_branch_7p_v4_b128_spl_3+1_tri0.15_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.18_stride1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                  --test_dir /home/tianlab/hengheng/reid \
                  --test_set DukeMTMC-reID \
                  --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                  --which_epoch 59 \
                  --batchsize 32;

python test_gen.py --gpu_ids 0 \
                  --test_dir /home/tianlab/hengheng/reid \
                  --test_set DukeMTMC-reID \
                  --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                  --which_epoch 99 \
                  --batchsize 32;

python test_gen.py --gpu_ids 0 \
                  --test_dir /home/tianlab/hengheng/reid \
                  --test_set DukeMTMC-reID \
                  --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_SGD \
                  --which_epoch 119 \
                  --batchsize 32;

#================
# evaluation
#================
# bash evaluate_gen.sh;
