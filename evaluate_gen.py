import scipy.io
import torch
import numpy as np
#import time
import os
import sys

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    mask = np.in1d(index, junk_index, invert=True) # numpy isin()
    index_mask = index[mask]
    return index_mask, index, good_index, junk_index


    #CMC_tmp = compute_mAP(index, good_index, junk_index)
    #return CMC_tmp

def rank_fusion(index_1, index_2):
    # mask = np.in1d(index_2[0:5], index_1[0:10], invert=True)
    # remain = index_2[0:5][mask]
    order_2 = []
    for id in index_1:
        idx_2 = np.argwhere(index_2 == id)
        order_2.extend(idx_2[0])
    order_2 = np.array(order_2)
    order_1 = np.array(list(range(len(index_1))))
    order_new = np.vstack([order_1, order_2])
    rank_median = np.median(order_new, axis=0)
    idx_median = np.argsort(rank_median)
    # print(len(index_1[idx_median]))
    return index_1[idx_median]


def compute_mAP(index_mask, index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    # mask = np.in1d(index, junk_index, invert=True) # numpy isin()
    # index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index_mask, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2 # area

    return ap, cmc

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = m_result['mquery_f']
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]

# CMC = torch.IntTensor(len(gallery_label)).zero_()
#
# ap = 0.0
# #print(query_label)
# for i in range(len(query_label)):
#     index_mask, index, good_index, junk_index = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#     # ap_tmp, CMC_tmp = compute_mAP(index_mask, index, good_index, junk_index)
#
#     mquery_index1 = np.argwhere(mquery_label==query_label[i])
#     mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
#     mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
#     mq = np.mean(mquery_feature[mquery_index,:], axis=0)
#
#     q_fusion = 1.25 * query_feature[i] +  0.25 * mq
#
#     m_index_mask, m_index, m_good_index, m_junk_index = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#     new_idx = rank_fusion(index_mask, m_index_mask)
#
#     ap_tmp, CMC_tmp = compute_mAP(new_idx, m_index, m_good_index, m_junk_index)
#
#     if CMC_tmp[0]==-1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     print(i, CMC_tmp[0])
#
# CMC = CMC.float()
# CMC = CMC/len(query_label) #average CMC
# print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
#
# sys.exit()
# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = np.mean(mquery_feature[mquery_index,:], axis=0)
        q_fusion = 1.25 * query_feature[i] +  0.25 * mq
        m_index_mask, m_index, m_good_index, m_junk_index = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        ap_tmp, CMC_tmp = compute_mAP(m_index_mask, m_index, m_good_index, m_junk_index)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
