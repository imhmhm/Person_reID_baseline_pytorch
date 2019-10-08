import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
# from torch.autograd import Variable
import pretrainedmodels
import sys

######################################################################


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--dropout--|--Linear--|

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=True, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        ########################
        # no bias in classifier
        # classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        #########################
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model


class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.0, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft

        ######################################
        # # freeze layers
        # fixed_names = []
        # for name, module in self.model._modules.items():
        #     if name == 'layer3':
        #         break;
        #     fixed_names.append(name)
        #     for param in module.parameters():
        #         param.requires_grad = False
        ######################################

        # self.classifier = ClassBlock(2048, class_num, droprate)

        ##### |--Linear--|--bn--|--relu--|--dropout--|--Linear--| #####
        # self.classifier = ClassBlock(2048, class_num, droprate, relu=True)

        ##### |--bn--|--Linear--| #####
        self.classifier = ClassBlock(2048, class_num, droprate, relu=False, linear=False)
        #######################
        # no shift(bias) in BN
        # self.classifier.add_block[0].bias.requires_grad_(False)
        #######################

        ##### |--bn--|--relu--|--Linear--| #####
        # self.classifier = ClassBlock(2048, class_num, droprate, relu=True, linear=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ft_net_feature(nn.Module):

    def __init__(self, class_num, droprate=0.0, stride=2):
        super(ft_net_feature, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # model_ft = models.resnet101(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft

        # self.classifier = ClassBlock(2048, class_num, droprate, relu=True)
        ####################### |--bn--|--Linear--| #######################
        self.classifier = ClassBlock(2048, class_num, droprate, relu=False, linear=False)
        # #==========================
        # # feature after bn
        # self.classifier = ClassBlock(2048, class_num, droprate, relu=False, linear=False, return_f=True)
        # #==========================
        #######################
        # no shift(bias) in BN
        # self.classifier.add_block[0].bias.requires_grad_(False)
        #######################

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)
        # #==========================
        # # feature after bn
        # x = x.view(x.size(0), -1)
        # x, feature = self.classifier(x)
        # #==========================
        return feature, x

#
# class ft_net_sub(nn.Module):
#
#     def __init__(self, class_num, droprate=0.0):
#         super(ft_net_sub, self).__init__()
#         model_ft = models.resnet50(pretrained=True)
#         # avg pooling to global pooling
#         model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model = model_ft
#         # self.classifier = ClassBlock(2048, class_num, droprate)
#         self.classifier_reid = ClassBlock(2048, class_num, droprate, relu=True)
#         self.classifier_gen = ClassBlock(2048, 2, droprate, relu=True)
#
#     def forward(self, x):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#         x = self.model.avgpool(x)
#         x = x.view(x.size(0), -1)
#         id = self.classifier_reid(x)
#         gen = self.classifier_gen(x)
#         return id, gen

# Define the DenseNet121-based Model


class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.0):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        # self.classifier = ClassBlock(1024, class_num, droprate, relu=True)
        ####################### |--bn--|--Linear--| #######################
        self.classifier = ClassBlock(1024, class_num, droprate, relu=False, linear=False)

    def forward(self, x):
        x = self.model.features(x)
        feature = x.view(x.size(0), x.size(1))
        x = self.classifier(feature)
        return feature, x


class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.0):
        super().__init__()
        model_name = 'nasnetalarge'
        model_ft = pretrainedmodels.__dict__[model_name](num_class=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).


class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.0):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)


class PCB(nn.Module):
    def __init__(self, class_num, part=6, droprate=0.0):
        super(PCB, self).__init__()

        self.part = part  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        ##===================================================
        ## pooling 6 parts feature
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

        # # no dropout
        # self.dropout = nn.Dropout(p=0.5)

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        ##===================================================
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            # setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=True, bnorm=True, num_bottleneck=256))

            ####################### |--bn--|--Linear--| #######################
            setattr(self, name, ClassBlock(2048, class_num, droprate, relu=False, linear=False))

        ##==========================================================================
        ## add whole feature
        self.avgpool_whole = nn.AdaptiveAvgPool2d((1, 1))

        ####################### |--bn--|--Linear--| #######################
        self.classifier_whole = ClassBlock(2048, class_num, droprate, relu=False, linear=False)
        ##===========================================================================

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        feat_bf_avgpool = self.model.layer4(x)
        x = self.avgpool(feat_bf_avgpool)


        ##=======================================================
        # feat_whole = self.avgpool_whole(feat_bf_avgpool)
        # feat_whole = feat_whole.view(feat_whole.size(0), -1)
        #
        # y_whole = self.classifier_whole(feat_whole)
        ##=======================================================

        ## no dropout
        # x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature (batchsize, 2048, 6)
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        ## sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #     y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])

        ##=====================
        # y.insert(0, y_whole)
        ##=====================

        return y


class PCB_test(nn.Module):
    def __init__(self, model, part=6):
        super(PCB_test, self).__init__()
        self.part = part

        ##===================================================
        ## feature after BN
        for i in range(self.part):

            name = 'classifier'+str(i)
            c = getattr(model, name)
            c.classifier = nn.Sequential()

        model.classifier_whole.classifier = nn.Sequential()

        self.model = model
        ##===================================================

        # self.model = model.model
        # self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        ## remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1, 1)
        # self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        #
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.avgpool(x)
        # y = x.view(x.size(0), x.size(1), x.size(2))

        ##===================================================
        ## feature after BN
        y = self.model(x)
        y = torch.stack(y, -1)
        ##===================================================
        ## in consistent with ft_net_feature
        return None, y


# debug model structure
# Here I left a simple forward function.
# Test the model, before you train it.
if __name__ == '__main__':

    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = torch.FloatTensor(8, 3, 256, 128)
    output = net(input)
    print('net output size:')
    print(output.shape)
