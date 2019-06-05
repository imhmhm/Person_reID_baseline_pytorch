# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
import os
import time
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from shutil import copyfile
import yaml
import argparse
import copy
import random
###############################################
from random_erasing import RandomErasing
from model import ft_net, ft_net_feature, ft_net_dense, PCB
from resnext import resnext50_32x4d_fc512 as resnext50
from warmup_scheduler import WarmupMultiStepLR
###############################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import torchvision
from torchvision import datasets, transforms
from torch.backends import cudnn
################################################
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, ConcatDataset  # Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
################################################
from hard_mine_triplet_loss import HardTripletLoss
from hard_mine_triplet_loss_mixup_v1 import TripletLoss_Mixup
# from hard_mine_multiple_loss import TripletLoss
# from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

version = torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir_1', default='/home/tianlab/hengheng/reid/Market/pytorch', type=str, help='training dir path')
parser.add_argument('--data_dir_2', default='/home/tianlab/hengheng/reid/DukeMTMC-reID/pytorch', type=str, help='generated training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_resnext', action='store_true', help='use resnext50')
parser.add_argument('--use_NAS', action='store_true', help='use NASnet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=120, type=int, help='epoch number')
parser.add_argument('--eps', default=0.4, type=float, help='label smoothing rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--warmup', action='store_true', help='use warmup lr_scheduler')
parser.add_argument('--lsr', action='store_true', help='use label smoothing')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--num_per_id', default=4, type=int, help='number of images per id in a batch')
parser.add_argument('--prop_label', default=2, type=int, help='ratio of real images per id in a batch')
parser.add_argument('--prop_unlabel', default=2, type=int, help='ratio of gen images per id in a batch')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

fp16 = opt.fp16
data_dir_1 = opt.data_dir_1
data_dir_2 = opt.data_dir_2
# gen_name = opt.gen_name
name = opt.name
assert(opt.num_per_id == (opt.prop_label + opt.prop_unlabel))
proportion = [opt.prop_label, opt.prop_unlabel]
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 128), interpolation=3),
        # transforms.Resize((288, 144), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + \
        [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

####################################################################
# dataset with generated smaples


class genDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, flag=0):
        super(genDataset, self).__init__(root, transform=transform,
                                         target_transform=target_transform, loader=loader)
        self.flag = flag

    def __getitem__(self, index):
        # if os.path.basename(self.root) == "train_all":
        #     self.flag = 0
        # elif os.path.basename(self.root) == "gen_train":
        #     self.flag = 1
        # else:
        #     self.flag = 0
        inputs, labels = super(genDataset, self).__getitem__(index)
        return inputs, labels, self.flag

###################################################################
# modified LSR_loss


class LSR_loss(nn.Module):
    # change target to range(0,750)
    def __init__(self, epsilon):
        super(LSR_loss, self).__init__()
        self.epsilon = epsilon

    # input is the prediction score(torch Variable) 32*752, target is the corresponding label,
    def forward(self, input, target, flg):
        # while flg means the flag(=0 for true data and 1 for generated data)  batchsize*1
        # print(type(input))
        # N defines the number of images, C defines channels,  K class in total
        assert(input.dim() <= 2)
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C

        # Max trick (output - max) for softmax
        # outputs.data  return the index of the biggest value in each row
        maxRow, _ = torch.max(input, 1)
        maxRow = maxRow.unsqueeze(1)
        input = input - maxRow

        epsilon = self.epsilon  # 0.4  # follow PT-GAN
        # epsilon = 0.3
        # epsilon = 1.0 # LSRO
        target = target.view(-1, 1)       # batchsize, 1
        flg = flg.view(-1, 1)
        # len=flg.size()[0]
        flos = F.log_softmax(input, dim=1)       # N, K = batchsize, 751
        flos = torch.sum(flos, 1) / flos.size(1)
        logpt = F.log_softmax(input, dim=1)      # size: batchsize ,751
        # print(logpt)
        logpt = logpt.gather(1, target)   # target N, 1
        logpt = logpt.view(-1)            # N*1 original loss
        flg = flg.view(-1)
        flg = flg.type(torch.cuda.FloatTensor)
        loss = -1 * logpt * (1 - epsilon * flg) - flos * epsilon * flg
        return loss.mean()

####################################################################
# sampler


class GenSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    proportion of real to gen: [real, gen]

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, proportion):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // 2 // self.num_instances
        self.real = self.num_instances
        # proportion[0] # labeled data
        # self.gen = proportion[1] # unlabeled data

        self.index_dic_real = defaultdict(list)
        # self.index_dic_gen = defaultdict(list)
        self.index_dic_gen = list()
        for index, (_, pid, flg) in enumerate(self.data_source):
            if flg == 0:
                self.index_dic_real[pid].append(index)
            elif flg == 1:
                # self.index_dic_gen[pid].append(index)
                self.index_dic_gen.append(index)
            else:
                raise ValueError

        self.pids_real = list(self.index_dic_real.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids_real:
            num_real = len(self.index_dic_real[pid])
            # num = len(idxs_real)
            if num_real < self.real:
                num_real = self.real
            self.length += num_real - num_real % self.real
        self.length = self.length * 2

    def __iter__(self):
        batch_idxs_dict_real = defaultdict(list)

        for pid in self.pids_real:

            idxs_real = copy.deepcopy(self.index_dic_real[pid])
            if len(idxs_real) < self.real:
                idxs_real = np.random.choice(idxs_real, size=self.real, replace=True)
            random.shuffle(idxs_real)

            batch_idxs_real = []

            for idx in idxs_real:
                batch_idxs_real.append(idx)
                if len(batch_idxs_real) == self.real:
                    batch_idxs_dict_real[pid].append(batch_idxs_real)
                    batch_idxs_real = []

        avai_pids = copy.deepcopy(self.pids_real)
        idxs_gen_list = copy.deepcopy(self.index_dic_gen)
        idxs_gen = np.random.choice(idxs_gen_list, size=self.batch_size//2, replace=False)

        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict_real[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict_real[pid]) == 0:
                    avai_pids.remove(pid)
            final_idxs.extend(idxs_gen)

        return iter(final_idxs)

    def __len__(self):
        return self.length

#####################################################################

###################################################################
# ramdom Sampler
# without controling ids and instances for each batch


class RandomGenSampler(Sampler):

    def __init__(self, data_source, cumulative_sizes, proportion):
        self.data_source = data_source
        self.cumulative_sizes = cumulative_sizes
        self.proportion = proportion

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        start = 0
        for i, seg in enumerate(self.cumulative_sizes):
            end = seg
            temp_idx_list = list(range(start, end))
            len_seg = len(
                temp_idx_list) // self.proportion[0] * self.proportion[i]
            temp = np.random.permutation(temp_idx_list)
            len_extend = len_seg - len(temp_idx_list)
            if len_extend > 0:
                seg_extend = np.random.choice(
                    temp_idx_list, size=len_extend, replace=True)
                temp.extend(seg_extend)
            elif len_extend < 0:
                temp = np.random.choice(
                    temp_idx_list, size=len_seg, replace=False)
            setattr(self, "seg{}".format(i), temp)
            start = end

        final = []
        for idx in range(self.cumulative_sizes[-1] // sum(self.proportion)):
            for j, p in enumerate(self.proportion):
                seg_list = getattr(self, "seg{}".format(j))
                final.extend(seg_list[idx:idx+p])

        return iter(final)

    def __len__(self):
        return self.num_samples

###################################################################


train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}

label_datasets = genDataset(os.path.join(data_dir_1, 'train' + train_all), data_transforms['train'], flag=0)
unlabel_datasets = genDataset(os.path.join(data_dir_2, 'train' + train_all), data_transforms['train'], flag=1)
image_datasets['train'] = ConcatDataset([label_datasets, unlabel_datasets])

cumulative_sizes = image_datasets['train'].cumulative_sizes

image_datasets['val'] = genDataset(os.path.join(data_dir_1, 'val'), data_transforms['val'], flag=0)

dataloaders = {}
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=opt.batchsize,
                                  sampler=GenSampler(image_datasets['train'], opt.batchsize, opt.num_per_id, proportion),
                                  num_workers=8, drop_last=True)  # 8 workers may work faster
dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=8,
                                shuffle=True, num_workers=0, drop_last=True)  # 8 workers may work faster

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = label_datasets.classes

# print(image_datasets['train'].cumulative_sizes)
# print(image_datasets['train'].__getitem__(12937))
# print(unlabel_datasets.class_to_idx)
# sys.exit()

use_gpu = torch.cuda.is_available()

# since = time.time()
# inputs, classes, flags = next(iter(dataloaders['train']))
# print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

########################################################################
# mixup
# dataloading


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_unlabel(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1

    batch_size = x.size()[0]
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)
    lam = 0.9
    mixed_x = lam * x[0:batch_size//2, :] + (1 - lam) * x[batch_size//2:, :]
    y_a = y[0:batch_size//2]
    y_b = y[batch_size//2:]
    return mixed_x, y_a, y_b, lam

# criterion


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

########################################################################
# train


def train_model(model, criterions, optimizer, scheduler, num_epochs=25, re_epoch=0):

    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    if opt.resume:
        # start = re_epoch
        start = 0
    else:
        start = 0
    for epoch in range(start, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_loss_xent = 0.0
            running_loss_htri = 0.0
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels, flags = data
                # print(flags)
                # sys.exit()
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda().detach()
                    labels = labels.cuda().detach()
                    flags = flags.cuda().detach()
                    # print(labels, flags)
                    # sys.exit()
                else:
                    inputs, labels, flags = inputs, labels, flags

                if opt.mixup:
                    # inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, use_cuda=use_gpu)
                    inputs, targets_a, targets_b, lam = mixup_data_unlabel(inputs, labels, alpha=0.2, use_cuda=use_gpu)
                    now_batch_size = inputs.shape[0]
                    # print(targets_a)
                    # print(targets_b)
                    # print(now_batch_size)
                    # sys.exit()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if opt.triplet:
                    features, outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if not opt.PCB:
                    _, preds = torch.max(outputs, 1)
                    if opt.mixup and opt.triplet:
                        # loss_xent = mixup_criterion(criterions['xent'], outputs, targets_a, targets_b, lam)
                        loss_xent = criterions['xent'](outputs, targets_a)
                        loss_htri = criterions['tri'](features, targets_a)
                        loss = 1.0 * loss_xent + 1.0 * loss_htri
                    elif opt.mixup:
                        # loss = mixup_criterion(criterions['xent'], outputs, targets_a, targets_b, lam)
                        loss = criterions['xent'](outputs, targets_a)
                    elif opt.triplet:
                        loss_xent = criterions['xent'](outputs, labels, flags)
                        loss_htri =  criterions['tri'](features, labels, flags, epoch)
                        loss = 1.0 * loss_xent + 1.0 * loss_htri
                    else:
                        loss = criterions['xent'](outputs, labels, flags)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + \
                        sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score, 1)

                    loss = criterions['xent'](part[0], labels, flags)
                    for i in range(num_part-1):
                        loss += criterions['xent'](part[i+1], labels, flags)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                # statistics
                # for the new version like 0.4.0, 0.5.0 and 1.0.0
                if int(version[0]) > 0 or int(version[2]) > 3:
                    running_loss += loss.item() * now_batch_size
                    if opt.triplet:
                        running_loss_xent += loss_xent.item() * now_batch_size
                        running_loss_htri += loss_htri.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                # running_corrects += float(torch.sum(preds == labels))
                print(preds)
                print(targets_a)
                running_corrects += float(torch.sum(preds == targets_a))
                print(running_corrects)
                sys.exit()


            epoch_loss = running_loss / dataset_sizes[phase]
            if opt.triplet:
                epoch_loss_xent = running_loss_xent / dataset_sizes[phase]
                epoch_loss_tri = running_loss_htri / dataset_sizes[phase]
                print('{} loss_xent: {:.4f} loss_tri: {:.4f}'.format(phase, epoch_loss_xent, epoch_loss_tri))
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve

x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))

######################################################################
# Save model


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

########################################################################
# load model


def load_network(network):
    epoch = 119
    load_name = 'ft_ResNet50_BT_b16x4_adam_warmup'
    save_path = os.path.join('./model', load_name, 'net_{}.pth'.format(epoch))
    network.load_state_dict(torch.load(save_path))
    return network, epoch

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


if opt.use_dense:
    model = ft_net_dense(len(class_names), opt.droprate)
elif opt.use_NAS:
    model = ft_net_NAS(len(class_names), opt.droprate)
elif (not opt.use_dense) and (not opt.use_NAS) and opt.triplet:
    model = ft_net_feature(len(class_names), opt.droprate, opt.stride)
else:
    model = ft_net(len(class_names), opt.droprate, opt.stride)

if opt.PCB:
    model = PCB(len(class_names))

if opt.resume:
    model, re_epoch = load_network(model)

opt.nclasses = len(class_names)

print(model)

criterions = {}
if opt.lsr:
    criterions['xent'] = LSR_loss(epsilon=opt.eps)
else:
    criterions['xent'] = nn.CrossEntropyLoss()

if opt.mixup:
    # criterions['tri'] = TripletLoss_Mixup(margin=0.3)
    criterions['tri'] = HardTripletLoss(margin=0.3)
else:
    criterions['tri'] = HardTripletLoss(margin=0.3)


if opt.adam:
    # BT: 0.00035
    optimizer_ft = optim.Adam(model.parameters(), 0.00035, weight_decay=5e-4)
# elif opt.use_resnext:
#     ignored_params = list(map(id, model.fc.parameters())) + \
#                      list(map(id, model.classifier.parameters()))
#     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
#     optimizer_ft = optim.SGD([
#              {'params': base_params, 'lr': 0.1*opt.lr},
#              {'params': model.fc.parameters(), 'lr': opt.lr},
#              {'params': model.classifier.parameters(), 'lr': opt.lr}
#          ], weight_decay=5e-4, momentum=0.9, nesterov=True)
elif not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters())) + \
                     list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (list(map(id, model.classifier0.parameters()))
                       + list(map(id, model.classifier1.parameters()))
                       + list(map(id, model.classifier2.parameters()))
                       + list(map(id, model.classifier3.parameters()))
                       + list(map(id, model.classifier4.parameters()))
                       + list(map(id, model.classifier5.parameters()))
                       # +list(map(id, model.classifier6.parameters() ))
                       # +list(map(id, model.classifier7.parameters() ))
                       )
    base_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             # {'params': model.classifier6.parameters(), 'lr': 0.01},
             # {'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
if opt.warmup and opt.adam:
    # BT: [40, 70]
    exp_lr_scheduler = WarmupMultiStepLR(optimizer_ft, milestones=[40, 70], gamma=0.1,
                                         warmup_factor=0.01, warmup_iters=10, warmup_method='linear')
elif opt.adam:
    # BT: [40, 70]
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 70], gamma=0.1)
else:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 80], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('./train_multi.py', dir_name+'/train_multi.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

if use_gpu:
    model = model.cuda()

if fp16:
    #model = network_to_half(model)
    #optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

model = train_model(model, criterions, optimizer_ft, exp_lr_scheduler,
                    num_epochs=opt.epoch)
