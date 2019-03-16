#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid/Market/pytorch \
                   --name ft_ResNet50_gen_e0.4_reid_branch_7p_v4_b128_spl_3+1 --which_epoch 99 --batchsize 32 \
                   # --triplet \
                   # --multi \
                   # --gen_query gen_query_7p_v4_4v \
                   #--PCB \
                   # DukeMTMC-reID / Market
# python evaluate.py;
