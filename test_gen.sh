#=============================================================================
python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 59 \
--batchsize 32;

python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 99 \
--batchsize 32;

python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set DukeMTMC-reID \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 119 \
--batchsize 32;
#
# #=============================================================================
python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set Market \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 59 \
--batchsize 32;

python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set Market \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 99 \
--batchsize 32;

python test_gen.py --gpu_ids 0 \
--test_dir /home/hmhm/reid \
--test_set Market \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--which_epoch 119 \
--batchsize 32;
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set Market \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
#
# # #====================================================================================================
# # #====================================================================================================
# #=============================================================================
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set DukeMTMC-reID \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
#
# # #====================================================================================================
# # #====================================================================================================
# #=============================================================================
# #=============================================================================
# # DukeMTMC-reID / Market / cuhk03-np
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
# #=============================================================================
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 59 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 99 \
# --batchsize 32;
#
# python test_gen.py --gpu_ids 0 \
# --test_dir /home/hmhm/reid \
# --test_set cuhk03-np \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --which_epoch 119 \
# --batchsize 32;
#
# #================
# # evaluation
# #================
# # bash evaluate_gen.sh;
