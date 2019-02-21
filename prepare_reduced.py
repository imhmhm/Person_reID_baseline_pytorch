import os
from shutil import copyfile
import numpy as np
import math

# You only need to change this line to your dataset download path
download_path = '/home/tianlab/hengheng/reid/Market'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch_third'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# #-----------------------------------------
# #query
# query_path = download_path + '/query'
# query_save_path = save_path + '/query'
# if not os.path.isdir(query_save_path):
#     os.mkdir(query_save_path)
#
# for root, dirs, files in os.walk(query_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = query_path + '/' + name
#         dst_path = query_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)
#
# #-----------------------------------------
# #multi-query
# query_path = download_path + '/gt_bbox'
# query_save_path = save_path + '/multi-query'
# if not os.path.isdir(query_save_path):
#     os.mkdir(query_save_path)
#
# for root, dirs, files in os.walk(query_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = query_path + '/' + name
#         dst_path = query_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)
#
# #-----------------------------------------
# #gallery
# gallery_path = download_path + '/bounding_box_test'
# gallery_save_path = save_path + '/gallery'
# if not os.path.isdir(gallery_save_path):
#     os.mkdir(gallery_save_path)
#
# for root, dirs, files in os.walk(gallery_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = gallery_path + '/' + name
#         dst_path = gallery_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#train_all
train_path = download_path + '/pytorch/train_all'
train_save_path = save_path + '/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    file_count = len(files)
    if file_count == 0:
        continue
    index = np.random.choice(file_count, size=math.ceil(file_count/3), replace=False)
    #index = np.random.choice(file_count, size=2, replace=False)
    files = np.array(files)
    for name in files[index]:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + ID[0] + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


#---------------------------------------
#train_val
train_path = download_path + '/pytorch/train_all'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    file_count = len(files)
    if file_count == 0:
        continue
    index = np.random.choice(file_count, size=math.ceil(file_count/3), replace=False)
    #index = np.random.choice(file_count, size=2, replace=False)
    files = np.array(files)
    for name in files[index]:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + ID[0] + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
