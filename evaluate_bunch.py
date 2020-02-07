import scipy.io
import torch
import numpy as np
# import time
import os
import sys
import argparse
import csv

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--test_sets', choices=['DukeMTMC-reID', 'Market', 'cuhk03-np', 'MSMT17_V2'], nargs='+', help='test sets name')
# DukeMTMC-reID / Market / cuhk03-np / MSMT17_V2
parser.add_argument('--which_epochs', default='last', type=str, nargs='+', help='0,1,2,3...or last')
# 59, 99, 119
parser.add_argument('--multi', action='store_true', help='evaluating multi-queries')
# parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
opt = parser.parse_args()
#######################################################################
# Evaluate

def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    ##### cosine similarity #####
    score = np.dot(gf, query)
    ## predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp

def evaluate_rk(idx, ql, qc, gl, gc):

    index = idx
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  # numpy isin()
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall*(old_precision + precision)/2  # area

    return ap, cmc

######################################################################

for test_set in opt.test_sets:
    for which_epoch in opt.which_epochs:
        feat_path = os.path.join('./model', opt.name, test_set, 'pytorch_result_{}.mat'.format(which_epoch))
        result = scipy.io.loadmat(feat_path)
        query_feature = result['query_f']
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]
        gallery_feature = result['gallery_f']
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        # index_path = os.path.join('./model', opt.name, test_set, 'return_index_sparse_k10.mat')
        # index = scipy.io.loadmat(index_path)['Idx_A']
        # # index_path = os.path.join('./model', opt.name, test_set, 'Idx_ori.mat')
        # # index = scipy.io.loadmat(index_path)['Idx_ori']
        # index = index - 1

        # multi = os.path.isfile('multi_query.mat')
        multi = opt.multi
        if multi:
            multi_path = os.path.join('./model', opt.name, test_set, 'multi_query_{}.mat'.format(which_epoch))
            m_result = scipy.io.loadmat(multi_path)
            mquery_feature = m_result['mquery_f']
            mquery_cam = m_result['mquery_cam'][0]
            mquery_label = m_result['mquery_label'][0]

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        # print(query_label)
        for i in range(len(query_label)):
            # index_i = np.squeeze(index[i, :])
            # ap_tmp, CMC_tmp = evaluate_rk(index_i, query_label[i], query_cam[i], gallery_label, gallery_cam)
            ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC/len(query_label)  # average CMC
        print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
        csv_path = os.path.join('./model', opt.name, test_set, 'result.csv')
        with open(csv_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['{}'.format(which_epoch), '{:f}'.format(CMC[0]), '{:f}'.format(CMC[4]),
                                 '{:f}'.format(CMC[9]), '{:f}'.format(ap/len(query_label))])

        # multiple-query
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        if multi:
            for i in range(len(query_label)):
                mquery_index1 = np.argwhere(mquery_label == query_label[i])
                mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
                mquery_index = np.intersect1d(mquery_index1, mquery_index2)
                mq = np.mean(mquery_feature[mquery_index, :], axis=0)
                ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                # print(i, CMC_tmp[0])
            CMC = CMC.float()
            CMC = CMC/len(query_label)  # average CMC
            print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
