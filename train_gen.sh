# python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_ResNet50_gan_e1_3v --train_all --batchsize 32
#python train_gan.py --gpu_ids 0 --name ft_DenseNet121_gan_half --train_all --batchsize 32 --use_dense

python train_gen.py --gpu_ids 0 \
                    --data_dir /home/tianlab/hengheng/reid/Market/pytorch \
                    --gen_name gen_train_reid_picked \
                    --name ft_ResNet50_mixup_picked --train_all --batchsize 32 \
                    --mixup \
                    --resume \
                    #--PCB \
