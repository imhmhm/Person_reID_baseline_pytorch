# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler

import numpy as np
from PIL import Image
import time
import os
import sys
import random
from collections import defaultdict
import copy
import math
from model import ft_net, ft_net_dense, PCB
from random_erasing import RandomErasing
import json
import tqdm
from shutil import copyfile

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/tianlab/hengheng/reid/Market/pytorch', type=str,
                    help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--num_per_id', default=4, type=int, help='number of images per id in a batch')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

# -------------------------------------------------------------------
# maket-duke
height = [0.0316, 0.05, 0.0533, 0.0383, 0.0449, 0.05, 0.05, 0.045]
width = [0.1211, 0.1308, 0.1275, 0.099, 0.1402, 0.13, 0.14, 0.1]
# maket/duke
ratio = [1.3631, 1.5455, 1.2857, 1.2542, 1.4264, 1.5455, 1.2857, 1.2333]  # w/h


def transform_market_to_duke(img):
    h_random = random.choice(height)
    w_random = random.choice(width)
    r_random = random.choice(ratio)
    h_new = round(256 * r_random)
    img = transforms.functional.resize(img, (h_new, 128))
    pad_random = (math.ceil(128 * w_random / 2), math.ceil(h_new * h_random / 2))
    img = transforms.functional.pad(img, padding=pad_random, padding_mode='reflect')
    # img.save("test_after.png")
    return img


# ----------------------------------------------------------------------
# add gausian noise
def transform_gaussian_noise(img):
    mean = 0.0
    std = 25.0
    img_np = np.array(img)
    noise_img_np = img_np + np.random.normal(mean, std, img_np.shape)
    noise_img_np = np.uint8(np.clip(noise_img_np, 0, 255))
    noise_img = Image.fromarray(noise_img_np)
    # noise_img.save("./test/test_after_{}.png".format(random.choice(range(100))), "PNG")
    return noise_img


# ------------------------------------------------------------------------

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),
    # transforms.Lambda(lambda x: transform_market_to_duke(x)),
    # transforms.Lambda(lambda x: transform_gaussian_noise(x)),
    transforms.Resize((288, 144), interpolation=3),
    # transforms.Pad(10),
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
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'
####################################################################
# sampler


class GenSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        # print('shuffle')
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:

            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs_select = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs_select)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return self.length

#####################################################################
# dataset


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'gen_train_7p_v1_x1'),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = dict()
# dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize, shuffle=True,
#                                                    num_workers=8, drop_last=True)
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize,
                                                   sampler=GenSampler(image_datasets['train'], opt.batchsize, opt.num_per_id),
                                                   num_workers=0, drop_last=True)

dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=True, num_workers=8,
                                                 drop_last=True)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
cls2idx = image_datasets['train'].class_to_idx

use_gpu = torch.cuda.is_available()

# since = time.time()
# inputs, classes = next(iter(dataloaders['train']))
# print(time.time() - since)
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


# criterion
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


########################################################################
# train


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    for epoch in range(num_epochs):
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
            # count = 0
            # Iterate over data.
            for data in tqdm.tqdm(dataloaders[phase]):
                # count += 1
                # print(count)
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # print(labels)
                    # sys.exit()
                else:
                    inputs, labels = inputs, labels

                if opt.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, use_cuda=use_gpu)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    if opt.mixup:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
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
# ---------------------------
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
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names), opt.droprate)
else:
    model = ft_net(len(class_names), opt.droprate)

if opt.PCB:
    model = PCB(len(class_names))

print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
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
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
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
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

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
copyfile('./train.py', dir_name + '/train.py')
copyfile('./model.py', dir_name + '/model.py')

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=100)
