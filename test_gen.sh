#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.5_SGD \
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
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.5_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.5_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_0.1_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_0.1_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_0.1_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri1.0_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri1.0_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri1.0_SGD\
                   --which_epoch 119 \
                   --batchsize 32;

#=============================================================================
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.1_SGD \
                   --which_epoch 59 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.1_SGD \
                   --which_epoch 99 \
                   --batchsize 32;

python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid \
                   --test_set DukeMTMC-reID \
                   --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.1_SGD \
                   --which_epoch 119 \
                   --batchsize 32;

#================
# evaluation
#================
bash evaluate_gen.sh;
