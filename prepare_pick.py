import os
import sys
import math
import numpy as np
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '/home/tianlab/hengheng/reid/Market'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# #-----------------------------------------
# #query
# query_path = download_path + '/query'
# query_save_path = download_path + '/pytorch/query'
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
# query_save_path = download_path + '/pytorch/multi-query'
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
# gallery_save_path = download_path + '/pytorch/gallery'
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
#
# #---------------------------------------
# #train_all
# train_path = download_path + '/bounding_box_train'
# train_save_path = download_path + '/pytorch/train_all'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)
#
#
# #---------------------------------------
# #train_val
# train_path = download_path + '/bounding_box_train'
# train_save_path = download_path + '/pytorch/train'
# val_save_path = download_path + '/pytorch/val'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#     os.mkdir(val_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#             dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# gen_train
train_path = download_path + '/pytorch/1501_train_7p_v4'
train_save_path = download_path + '/pytorch/1501_train_7p_v4_x1'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    f = np.array(files)
    f = f[[x[-5] != '0' for x in f]]
    l = len(f)
    if l == 0:
        continue

    f_list = np.random.choice(l, math.ceil(l / 3), replace=False)
    # for name in f:
    for name in f[f_list]:
        if not name[-3:] == 'png':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + ID[0] + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


# for root, dirs, files in os.walk(train_path, topdown=True):
#     f = np.array(files)
#     f = f[[x[-5] != '0' for x in f]]
#     len_views = len(f)
#     print(len_views)
#     len_ori = len_views // 9
#     #print(len_ori)
#     if len_ori >= 12:
#         dst_path = train_save_path + '/' + root[-4:]
#         os.mkdir(dst_path)
#     elif len_ori < 12 and len_views + len_ori >=12:
#         f_list = np.random.choice(len_views, (12-len_ori), replace=False)
#         for name in f[f_list]:
#             if not name[-3:]=='png':
#                 continue
#             # if not name[-4]=='0':
#             #     continue
#             ID  = name.split('_')
#             src_path = train_path + '/' + ID[0] + '/' + name
#             dst_path = train_save_path + '/' + ID[0]
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             copyfile(src_path, dst_path + '/' + name)
#     else:
#         for name in f:
#             if not name[-3:]=='png':
#                 continue
#             ID  = name.split('_')
#             src_path = train_path + '/' + ID[0] + '/' + name
#             dst_path = train_save_path + '/' + ID[0]
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             copyfile(src_path, dst_path + '/' + name)
#

# ---------------------------------------
# gen_query
# train_path = download_path + '/gen/test_gen_seg'
# train_save_path = download_path + '/pytorch/gen_query'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='png':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)


#---------------------------------------
