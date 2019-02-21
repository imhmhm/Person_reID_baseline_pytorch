#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid/Market/pytorch \
                   --name ft_ResNet50_gen_e0.4_reid_branch_7p_v1_b128_spl_2+2 --which_epoch 99 --batchsize 32;
                   # --multi \
                   #--PCB \
                   # DukeMTMC-reID / Market
# python evaluate.py;
