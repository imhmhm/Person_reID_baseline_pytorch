#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid/Market/pytorch \
                   --name ft_ResNet50_gen_e0.4_2v_norelu --which_epoch 59 --batchsize 32
