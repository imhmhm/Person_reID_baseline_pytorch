# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

# gen_train_7p_v4_x1 \

# DukeMTMC-reID / Market

python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name 1501_train_7p_v4_x1 \
                    --lsr \
                    --eps_gen 0.4 \
                    --eps_real 0.0 \
                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.5_SGD \
                    --train_all \
                    --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --wt_tri 0.5 \
                    --lr 0.01 \
                    --triplet;
                    # --adam \
                    # --fp16 \
                    # --erasing_p 0.5 \
                    # --use_dense \
                    # --PCB \
                    # --use_resnext \
                    #--mixup \
                    #--resume \

python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name 1501_train_7p_v4_x1 \
                    --lsr \
                    --eps_gen 0.4 \
                    --eps_real 0.0 \
                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                    --train_all \
                    --batchsize 64 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --wt_tri 0.5 \
                    --lr 0.01 \
                    --triplet;
                    # --adam \


python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name 1501_train_7p_v4_x1 \
                    --lsr \
                    --eps_gen 0.4 \
                    --eps_real 0.1 \
                    --name market_ResNet50_gen_e0.4_0.1_branch_7p_v4_b64_spl_3+1_tri0.5_SGD \
                    --train_all \
                    --batchsize 64 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --wt_tri 0.5 \
                    --lr 0.01 \
                    --triplet;


python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name 1501_train_7p_v4_x1 \
                    --lsr \
                    --eps_gen 0.4 \
                    --eps_real 0.0 \
                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri1.0_SGD \
                    --train_all \
                    --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --wt_tri 1.0 \
                    --lr 0.01 \
                    --triplet;

python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name 1501_train_7p_v4_x1 \
                    --lsr \
                    --eps_gen 0.4 \
                    --eps_real 0.0 \
                    --name market_ResNet50_gen_e0.4_branch_7p_v4_b128_spl_3+1_tri0.1_SGD \
                    --train_all \
                    --batchsize 128 \
                    --num_per_id 4 \
                    --prop_real 3 \
                    --prop_gen 1 \
                    --wt_tri 0.1 \
                    --lr 0.01 \
                    --triplet;

#=============
# test
#=============
bash test_gen.sh
