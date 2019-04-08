# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from market_test import Market1501
from duke_test import Duke
import time
import os
import sys
import scipy.io
from model import ft_net, ft_net_feature, ft_net_dense, PCB, PCB_test

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--gen_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='/home/tianlab/hengheng/reid/Market/pytorch', type=str, help='dataset dir')
parser.add_argument('--gen_query', default='/home/tianlab/hengheng/reid/Market/pytorch/gen_query', type=str, help='gen multi_query')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='real model path')
parser.add_argument('--name_gen', default='ft_ResNet50', type=str, help='gen model path')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--triplet', action='store_true', help='use triplet')
parser.add_argument('--multi', action='store_true', help='use multiple query')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
name_gen = opt.name_gen
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        #transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.ToTensor()(crop)
    #      for crop in crops]
    # )),
        #transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
    #       for crop in crops]
    # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir

# if opt.multi:
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query', opt.gen_query]}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                                   shuffle=False, num_workers=16) for x in ['gallery', 'query', opt.gen_query]}
# else:
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                                   shuffle=False, num_workers=16) for x in ['gallery', 'query']}
#
# image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'), data_transforms)
# class_names = image_datasets['query'].classes
# train_class_names = image_datasets['train'].classes
image_datasets = {}
dataloaders = {}
#image_datasets['gallery'] = Market1501(amount_gen=4, transform=data_transforms, mode='gallery')
image_datasets['gallery'] = Duke(amount_gen=5, transform=data_transforms, mode='gallery')
dataloaders['gallery'] = torch.utils.data.DataLoader(image_datasets['gallery'], batch_size=opt.batchsize, shuffle=False, num_workers=0)
#image_datasets['query'] = Market1501(amount_gen=4, transform=data_transforms, mode='query')
image_datasets['query'] = Duke(amount_gen=5, transform=data_transforms, mode='query')
dataloaders['query'] = torch.utils.data.DataLoader(image_datasets['query'], batch_size=opt.batchsize, shuffle=False, num_workers=0)

# print(len(image_datasets['gallery']))
# print(repr(image_datasets['gallery']))
# sys.exit()
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------


def load_network(network, name, epoch):
    save_path = os.path.join('./model', name, 'net_%s.pth' % epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, model_gen, dataloaders):
    # features = torch.FloatTensor()
    gen_features = torch.FloatTensor()
    cams = torch.LongTensor()
    labels = torch.LongTensor()
    imids = torch.LongTensor()
    count = 0
    for data in dataloaders:
        # img, label = data
        gen_img_0, gen_img_1, gen_img_2, gen_img_3, gen_img_4, pid, camid, imid = data
        n, c, h, w = gen_img_0.size()
        count += n
        print(count)

        # if opt.use_dense:
        #     ff = torch.FloatTensor(n, 1024).zero_()
        # else:
        #     ff = torch.FloatTensor(n, 2048).zero_()
        #     # ff = torch.FloatTensor(n, 512).zero_()
        # if opt.PCB:
        #     ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        # for i in range(2):
        #     if(i == 1):
        #         img = fliplr(img)
        #     input_img = Variable(img.cuda())
        #     if opt.triplet:
        #         outputs, _ = model_gen(input_img)
        #     elif opt.use_dense:
        #         outputs, _ = model_gen(input_img)
        #     else:
        #         outputs = model_gen(input_img)
        #     f = outputs.data.cpu()
        #     ff = ff+f
        # # norm feature
        # if opt.PCB:
        #     # feature size (n,2048,6)
        #     # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        #     # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        #     fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        #     ff = ff.div(fnorm.expand_as(ff))
        #     ff = ff.view(ff.size(0), -1)
        # else:
        #     fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        #     ff = ff.div(fnorm.expand_as(ff))
        # features = torch.cat((features, ff), 0)

        gen_features_combined = []
        for gen_img in [gen_img_0, gen_img_1, gen_img_2, gen_img_3, gen_img_4]:
            if opt.use_dense:
                ff_gen = torch.FloatTensor(n, 1024).zero_()
            else:
                ff_gen = torch.FloatTensor(n, 2048).zero_()
                # ff = torch.FloatTensor(n, 512).zero_()
            if opt.PCB:
                ff_gen = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
            for i in range(2):
                if(i == 1):
                    gen_img = fliplr(gen_img)
                input_gen_img = Variable(gen_img.cuda())
                if opt.triplet:
                    outputs_gen, _ = model_gen(input_gen_img)
                elif opt.use_dense:
                    outputs_gen, _ = model_gen(input_gen_img)
                else:
                    outputs_gen = model_gen(input_gen_img)
                f_gen = outputs_gen.data.cpu()
                ff_gen = ff_gen + f_gen
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff_gen, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff_gen = ff_gen.div(fnorm.expand_as(ff_gen))
                ff_gen = ff_gen.view(ff_gen.size(0), -1) # 2048*6
            else:
                fnorm = torch.norm(ff_gen, p=2, dim=1, keepdim=True)
                ff_gen = ff_gen.div(fnorm.expand_as(ff_gen))
                gen_features_combined.append(ff_gen)
        gen_features_stack = torch.stack(gen_features_combined, -1)
        # mean of N gen features
        gen_features_final = torch.mean(gen_features_stack, -1)

        gen_features = torch.cat((gen_features, gen_features_final), 0)

        cams = torch.cat((cams, camid), 0)
        labels = torch.cat((labels, pid), 0)
        imids = torch.cat((imids, imid), 0)
        # print(pid, camid)
    return gen_features, cams, labels, imids


# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         # filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2] == '-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels
#
#
# gallery_path = image_datasets['gallery'].imgs
# query_path = image_datasets['query'].imgs
#
# gallery_cam, gallery_label = get_id(gallery_path)
# query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets[opt.gen_query].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751)
elif opt.triplet:
    model_structure = ft_net_feature(751)
else:
    model_structure = ft_net(751)

if opt.PCB:
    model_structure = PCB(751)

model = load_network(model_structure, opt.name, opt.which_epoch)
model_gen = load_network(model_structure, opt.name_gen, opt.gen_epoch)


# Remove the final fc layer and classifier layer
if opt.PCB:
    model = PCB_test(model)
    model_gen = PCB_test(model_gen)
else:
    if opt.triplet:
        model = model
    else:
        model.model.fc = nn.Sequential()
        model_gen.model.fc = nn.Sequential()
        model.classifier = nn.Sequential()
        model_gen.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
model_gen = model_gen.eval()
if use_gpu:
    model = model.cuda()
    model_gen = model_gen.cuda()

# Extract feature
with torch.no_grad():
    gen_gallery_feature, gallery_cam, gallery_label, gallery_imid = extract_feature(model, model_gen, dataloaders['gallery'])
    gen_query_feature, query_cam, query_label, query_imid = extract_feature(model, model_gen, dataloaders['query'])

    if opt.multi:
        mquery_feature = extract_feature(model, model_gen, dataloaders[opt.gen_query])

# Save to Matlab for check
# result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label.numpy(), 'gallery_cam': gallery_cam.numpy(),
#           'query_f': query_feature.numpy(), 'query_label': query_label.numpy(), 'query_cam': query_cam.numpy()}
# scipy.io.savemat('pytorch_result.mat', result)
#
gen_result = {'gen_gallery_f': gen_gallery_feature.numpy(), 'gallery_label': gallery_label.numpy(), 'gallery_cam': gallery_cam.numpy(),
              'gallery_imid': gallery_imid.numpy(),
              'gen_query_f': gen_query_feature.numpy(), 'query_label': query_label.numpy(), 'query_cam': query_cam.numpy(),
              'query_imid': query_imid.numpy(),}
scipy.io.savemat('pytorch_gen_result.mat', gen_result)

if opt.multi:
    result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
    scipy.io.savemat('multi_query.mat', result)
