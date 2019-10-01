# DukeMTMC-reID / Market / cuhk03-np / MSMT17_V2
#===============================================
python test_bunch.py \
--name ft_ResNet50_b16x4_stride1_lsr_tri1.0_warmup_RE_adam_msmt17 \
--test_dir /home/hmhm/reid \
--test_sets MSMT17_V2 Market DukeMTMC-reID cuhk03-np \
--which_epochs 59 99 119;

#===============================================
python test_bunch.py \
--name ft_ResNet50_b16x4_stride1_SGD_msmt17 \
--test_dir /home/hmhm/reid \
--test_sets MSMT17_V2 Market DukeMTMC-reID cuhk03-np \
--which_epochs 59 99 119;

#===============================================
python test_bunch.py \
--name ft_ResNet50_b16x4_stride1_lsr_tri1.0_adam_msmt17 \
--test_dir /home/hmhm/reid \
--test_sets MSMT17_V2 Market DukeMTMC-reID cuhk03-np \
--which_epochs 59 99 119;

#===============================================
python test_bunch.py \
--name ft_ResNet50_b16x4_stride1_tri1.0_adam_msmt17 \
--test_dir /home/hmhm/reid \
--test_sets MSMT17_V2 Market DukeMTMC-reID cuhk03-np \
--which_epochs 59 99 119;

#===============================================
python test_bunch.py \
--name ft_ResNet50_b16x4_tri1.0_adam_msmt17 \
--test_dir /home/hmhm/reid \
--test_sets MSMT17_V2 Market DukeMTMC-reID cuhk03-np \
--which_epochs 59 99 119;

#========================
#========================
# bash evaluate_bunch.sh;
