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
parser.add_argument('--test_set', default='Market', type=str, help='test set name')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='evaluating multi-queries')
opt = parser.parse_args()
#######################################################################
# Evaluate

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    # indices = indices[:,::-1]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


######################################################################
feat_path = os.path.join('./model', opt.name, opt.test_set, 'pytorch_result_{}.mat'.format(opt.which_epoch))
result = scipy.io.loadmat(feat_path)
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

# multi = os.path.isfile('multi_query.mat')
multi = opt.multi
if multi:
    multi_path = os.path.join('./model', opt.name, opt.test_set, 'multi_query_{}.mat'.format(opt.which_epoch))
    m_result = scipy.io.loadmat(multi_path)
    mquery_feature = m_result['mquery_f']
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]

# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# print(query_label)
m, n = query_feature.shape[0], gallery_feature.shape[0]
distmat = np.broadcast_to(np.power(query_feature, 2).sum(axis=1, keepdims=True), (m, n)) + \
          np.broadcast_to(np.power(gallery_feature, 2).sum(axis=1, keepdims=True), (n, m)).T
distmat = distmat - 2 * np.dot(query_feature, gallery_feature.T)
# distmat = np.dot(query_feature, gallery_feature.T)
CMC, ap = eval_func(distmat, query_label, gallery_label, query_cam, gallery_cam, max_rank=100)
# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
#     if CMC_tmp[0] == -1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     print(i, CMC_tmp[0])

# CMC = CMC.float()
# CMC = CMC/len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap))
csv_path = os.path.join('./model', opt.name, opt.test_set, 'result.csv')
with open(csv_path, 'a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['{}'.format(opt.which_epoch), '{:f}'.format(CMC[0]), '{:f}'.format(CMC[4]),
                         '{:f}'.format(CMC[9]), '{:f}'.format(ap)])

# multiple-query
# CMC = torch.IntTensor(len(gallery_label)).zero_()
CMC = np.zeros(len(gallery_label))
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
    CMC = CMC/len(query_label) # average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
