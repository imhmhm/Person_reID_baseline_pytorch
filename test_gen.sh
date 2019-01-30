#python test_gen.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch 59 --multi
python test_gen.py --gpu_ids 0 \
                   --test_dir /home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch \
                   --name ft_ResNet50_mixup_picked --which_epoch 59 --batchsize 32 \
                   #--PCB \
                   # DukeMTMC-reID / Market
