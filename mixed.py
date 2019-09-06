import numpy as np
import torch
from opt import opt

########################################################################
# mixed data augmentation methods
########################################################################

#===============
# mixup
#===============
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

#================================
# mixup for data sampler
#================================
def mixup_data_metric(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        # index = torch.randperm(batch_size).cuda()

        # N identities
        index_N_1 = torch.randperm(opt.batchsize // opt.num_per_id).cuda()
        index_N_2 = torch.randperm(opt.batchsize // opt.num_per_id).cuda()
        index_K_1 = torch.randperm(4).cuda()
        index_K_2 = torch.randperm(4).cuda()
        index_random_id_1 = torch.zeros(batch_size, dtype=torch.int64).cuda()
        index_random_id_2 = torch.zeros(batch_size, dtype=torch.int64).cuda()
        for i in range(batch_size):
            # K instances
            index_random_id_1[i] = index_N_1[i // 4] * 4 + index_K_1[i % 4]
            index_random_id_2[i] = index_N_2[i // 4] * 4 + index_K_2[i % 4]
    else:
        index = torch.randperm(batch_size)

    # mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]

    mixed_x_same_id_1 = lam * x + (1 - lam) * x[index_random_id_1, :]
    y_b_same_id_1 = y[index_random_id_1]
    y_a = y

    mixed_x_same_id_2 = lam * x + (1 - lam) * x[index_random_id_2, :]
    y_b_same_id_2 = y[index_random_id_2]

    mixed_x_double = torch.cat((mixed_x_same_id_1, mixed_x_same_id_2))
    y_a_double = torch.cat((y_a, y_a))
    y_b_double = torch.cat((y_b_same_id_1, y_b_same_id_2))

    # print(y_a_double)
    # print(y_b_double)
    # print(lam)
    # sys.exit()

    return mixed_x_double, y_a_double, y_b_double, lam
    # return mixed_x_same_id, y_a, y_b_same_id, lam

#================================
# vertical concatenation
#================================
def stitch_data(x, y, alpha=3.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, c, h, w = x.size()
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    stitch_x = torch.cat((x[:,:,0:round(h*lam),:], x[index,:,round(h*lam):h,:]), dim=2)
    y_a, y_b = y, y[index]
    return stitch_x, y_a, y_b, lam

#=============================================
# vertical concatenation for data sampler
#=============================================
def stitch_data_metric(x, y, alpha=3.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, c, h, w = x.size()
    if use_cuda:
        # index = torch.randperm(batch_size).cuda()
        # N identities
        index_N_1 = torch.randperm(opt.batchsize // opt.num_per_id).cuda()
        index_K_1 = torch.randperm(4).cuda()
        index_random_id_1 = torch.zeros(batch_size, dtype=torch.int64).cuda()

        index_N_2 = torch.randperm(opt.batchsize // opt.num_per_id).cuda()
        index_K_2 = torch.randperm(4).cuda()
        index_random_id_2 = torch.zeros(batch_size, dtype=torch.int64).cuda()
        for i in range(batch_size):
            # K instances
            index_random_id_1[i] = index_N_1[i // 4] * 4 + index_K_1[i % 4]
            index_random_id_2[i] = index_N_2[i // 4] * 4 + index_K_2[i % 4]
    else:
        index = torch.randperm(batch_size)

    # stitch_x = torch.cat((x[:,:,0:round(h*lam),:], x[index,:,round(h*lam):h,:]), dim=2)
    # y_a, y_b = y, y[index]

    stitch_x_same_id_1 = torch.cat((x[:,:,0:round(h*lam),:], x[index_random_id_1,:,round(h*lam):h,:]), dim=2)
    y_b_same_id_1 = y[index_random_id_1]
    y_a = y

    stitch_x_same_id_2 = torch.cat((x[:,:,0:round(h*lam),:], x[index_random_id_2,:,round(h*lam):h,:]), dim=2)
    y_b_same_id_2 = y[index_random_id_2]

    stitch_x_double = torch.cat((stitch_x_same_id_1, stitch_x_same_id_2))
    y_a_double = torch.cat((y_a, y_a))
    y_b_double = torch.cat((y_b_same_id_1, y_b_same_id_2))

    return stitch_x_double, y_a_double, y_b_double, lam
