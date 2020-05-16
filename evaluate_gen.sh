#========================================================================
python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set Market \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set Market \
--which_epoch 99;

python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set Market \
--which_epoch 119;

#========================================================================
python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set DukeMTMC-reID \
--which_epoch 59;

# # DukeMTMC-reID / Market

python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set DukeMTMC-reID \
--which_epoch 99;

python evaluate_gen.py \
--name market_ResNet50_gen_e0.4_7p_v4_x1_b8_spl_3+1_tri0.3_stride1_SGD \
--test_set DukeMTMC-reID \
--which_epoch 119;

# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set Market \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set Market \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set Market \
# --which_epoch 119;
# # # # #========================================================================================
# # # # #========================================================================================
# # #========================================================================
# # #========================================================================
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name duke_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 59;
#
# # # DukeMTMC-reID / Market
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name market_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set DukeMTMC-reID \
# --which_epoch 119;
#
# ##=======================================================================
# ##=======================================================================
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# # DukeMTMC-reID / Market / cuhk03-np
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.15_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# # DukeMTMC-reID / Market / cuhk03-np
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.1_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 119;
#
# #========================================================================
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 59;
#
# # DukeMTMC-reID / Market / cuhk03-np
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 99;
#
# python evaluate_gen.py \
# --name cuhk_ResNet101_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.3_stride1_SGD \
# --test_set cuhk03-np \
# --which_epoch 119;
