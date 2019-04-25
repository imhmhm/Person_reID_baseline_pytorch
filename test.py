# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import torchvision
from torchvision import datasets, transforms
# import time
import numpy as np
import os
import sys
import scipy.io
import yaml
from model import ft_net, ft_net_dense, PCB, PCB_test, ft_net_feature

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='/home/hmhm/reid', type=str, help='test set base path')
parser.add_argument('--test_set', default='Market', type=str, help='test set name')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')

opt = parser.parse_args()
###########################
#### load config ####
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'nclasses' in config:
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751
############################

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
test_set = opt.test_set

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
# We will use torchvision and torch.utils.data packages for loading the data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        # transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Ten Crop
        # transforms.TenCrop(224),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop)
        #                                              for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        #                                              for crop in crops]))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


#data_dir = test_dir
data_dir = os.path.join(test_dir, test_set, 'pytorch')

if opt.multi:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query', 'multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}

image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'), data_transforms)
class_names = image_datasets['query'].classes
train_class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------


def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
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


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()  # 2048 / 512
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = img.cuda()
            outputs = model(input_img)
            f = outputs.cpu().float()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride=opt.stride)

if opt.PCB:
    model_structure = PCB(opt.nclasses)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if not opt.PCB:
    model.model.fc = nn.Sequential()
    ##### feature after avgpool #####
    # model.classifier = nn.Sequential()
    ##### feature after BN #####
    # model.classifier.add_block[0] = nn.Sequential()
    model.classifier.add_block[1] = nn.Sequential()
    model.classifier.classifier = nn.Sequential()
    # print(model.classifier)
    # sys.exit()
else:
    model = PCB_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders['multi-query'])

# Save to Matlab for check
feat_dir = os.path.join('./model', name, test_set)
if not os.path.isdir(feat_dir):
    os.makedirs(feat_dir)

result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
feat_path = os.path.join('./model', name, test_set, 'pytorch_result_{}.mat'.format(opt.which_epoch))
scipy.io.savemat(feat_path, result)
if opt.multi:
    result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
    multi_path = os.path.join('./model', name, test_set, 'multi_query_{}.mat'.format(opt.which_epoch))
    scipy.io.savemat(multi_path, result)
