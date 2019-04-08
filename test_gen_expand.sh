#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen_expand.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
                   --name  ft_ResNet50_b128_spl_32x4_re0.5 \
                   --name_gen ft_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri_fp16_v2 \
                   --which_epoch 99 --gen_epoch 99 --batchsize 128 \
                   # --triplet \
                   # --use_dense \
                   # --PCB \
                   # --multi \
                   # --gen_query gen_query_7p_v4_3v \

                   # DukeMTMC-reID / Market
# python evaluate.py;
