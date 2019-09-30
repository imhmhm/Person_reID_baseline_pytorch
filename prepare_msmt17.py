import os
from shutil import copyfile, copytree

# You only need to change this line to your dataset download path
download_path = '/home/tianlab/hengheng/reid/MSMT17_V2'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

#-----------------------------------------
#query
query_path = download_path + '/mask_test_v2'
query_save_path = save_path + '/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

f_query = open(os.path.join(download_path, 'list_query.txt'), 'r')

list_query = f_query.readlines()

for f in list_query:
    name = f.split(' ')
    src_path = os.path.join(query_path, name[0])
    dst_path = os.path.join(query_save_path, name[0][0:4])
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, os.path.join(query_save_path, name[0]))

f_query.close()

#-----------------------------------------
#gallery
gallery_path = download_path + '/mask_test_v2'
gallery_save_path = save_path + '/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

with open(os.path.join(download_path, 'list_gallery.txt'), 'r') as f_gallery:
    list_gallery = f_gallery.readlines()

    for f in list_gallery:
        name = f.split(' ')
        src_path = os.path.join(gallery_path, name[0])
        dst_path = os.path.join(gallery_save_path, name[0][0:4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(gallery_save_path, name[0]))

#---------------------------------------
#train_all
train_path = download_path + '/mask_train_v2'
train_save_path = save_path + '/train_all'
copytree(train_path, train_save_path)

#---------------------------------------
#train_val
train_path = download_path + '/mask_train_v2'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

with open(os.path.join(download_path, 'list_train.txt'), 'r') as f_train:
    list_train = f_train.readlines()

    for f in list_train:
        name = f.split(' ')
        src_path = os.path.join(train_path, name[0])
        dst_path = os.path.join(train_save_path, name[0][0:4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(train_save_path, name[0]))

with open(os.path.join(download_path, 'list_val.txt'), 'r') as f_val:
    list_val = f_val.readlines()

    for f in list_val:
        name = f.split(' ')
        src_path = os.path.join(train_path, name[0])
        dst_path = os.path.join(val_save_path, name[0][0:4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(val_save_path, name[0]))
#---------------------------------------
