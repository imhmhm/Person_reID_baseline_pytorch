import scipy.io
import torch
import numpy as np
# import time
import os
import sys

#######################################################################
# Evaluate


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    # predict index
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

# def evaluate_rerank(score,ql,qc,gl,gc):
#     index = np.argsort(score)  #from small to large
#     #index = index[::-1]
#     # good index
#     query_index = np.argwhere(gl==ql)
#     camera_index = np.argwhere(gc==qc)
#
#     good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
#     junk_index1 = np.argwhere(gl==-1)
#     junk_index2 = np.intersect1d(query_index, camera_index)
#     junk_index = np.append(junk_index2, junk_index1) #.flatten())
#
#     CMC_tmp = compute_mAP(index, good_index, junk_index)
#     return CMC_tmp

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
result = scipy.io.loadmat('pytorch_result.mat')

query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
query_imid = result['query_imid'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
gallery_imid = result['gallery_imid'][0]

gen_result = scipy.io.loadmat('pytorch_gen_result.mat')

gen_query_feature = gen_result['gen_query_f']
gen_query_cam = gen_result['query_cam'][0]
gen_query_label = gen_result['query_label'][0]
gen_query_imid = gen_result['query_imid'][0]
gen_gallery_feature = gen_result['gen_gallery_f']
gen_gallery_cam = gen_result['gallery_cam'][0]
gen_gallery_label = gen_result['gallery_label'][0]
gen_gallery_imid = gen_result['gallery_imid'][0]

print(gallery_label)
print(gallery_feature.shape)

# multi = os.path.isfile('multi_query.mat')

# if multi:
#     m_result = scipy.io.loadmat('multi_query.mat')
#     mquery_feature = m_result['mquery_f']
#     mquery_cam = m_result['mquery_cam'][0]
#     mquery_label = m_result['mquery_label'][0]

# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# # print(query_label)
# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
#     if CMC_tmp[0] == -1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     print(i, CMC_tmp[0])
#
# CMC = CMC.float()
# CMC = CMC/len(query_label)  # average CMC
# print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
# file = open('result_reid.txt', 'a')
# file.writelines('%f,  %f,  %f,  %f \n' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
# file.close()

# sys.exit()
#
# CMC = torch.IntTensor(len(gen_gallery_label)).zero_()
# ap = 0.0
# for i in range(len(gen_query_label)):
#     ap_tmp, CMC_tmp = evaluate(gen_query_feature[i], gen_query_label[i], gen_query_cam[i], gen_gallery_feature, gen_gallery_label, gen_gallery_cam)
#     if CMC_tmp[0] == -1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     print(i, CMC_tmp[0])
#
# CMC = CMC.float()
# CMC = CMC/len(gen_query_label)  # average CMC
# print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(gen_query_label)))
# file = open('result_reid.txt', 'a')
# file.writelines('%f,  %f,  %f,  %f \n' % (CMC[0], CMC[4], CMC[9], ap/len(gen_query_label)))
# file.close()

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0


multi = True
if multi:

    gallery_feature_new = np.zeros_like(gallery_feature)

    for i in range(len(gallery_label)):
        gen_gallery_index1 = np.argwhere(gen_gallery_imid == gallery_imid[i])
        # gen_gallery_index2 = np.argwhere(gen_gallery_cam == gallery_cam[i])
        # gen_gallery_index3 = np.argwhere(gen_gallery_label == gallery_label[i])
        # gen_gallery_index = np.intersect1d(gen_gallery_index1, gen_gallery_index3)
        if len(gen_gallery_index1) != 0:
            gallery_feature_new[i] = np.maximum(gallery_feature[i], 0.1 * gen_gallery_feature[gen_gallery_index1])#[0][0])
            # print(gallery_feature_new[i])
            # print(gallery_feature[i])
            # print(gen_gallery_feature[gen_gallery_index1])
            # sys.exit()
        else:
            gallery_feature_new[i] = gallery_feature[i]

    query_feature_new = np.zeros_like(query_feature)
    for i in range(len(query_label)):
        gen_query_index1 = np.argwhere(gen_query_imid == query_imid[i])
        # gen_query_index2 = np.argwhere(gen_query_cam == query_cam[i])
        # gen_query_index3 = np.argwhere(gen_query_label == query_label[i])
        # gen_query_index = np.intersect1d(gen_query_index1, gen_query_index3)
        if len(gen_query_index1) != 0:
            query_feature_new[i] = np.maximum(query_feature[i], 0.1 * gen_query_feature[gen_query_index1])
        else:
            query_feature_new[i] = query_feature[i]
        # print(query_feature_new[i])
        # print(gen_query_feature[gen_query_index1])
        # print(query_feature[i])
        # sys.exit()
        # mq = np.mean(mquery_feature[mquery_index, :], axis=0)
        # mq = np.maximum(query_feature[i], gen_query_feature[i])
        # mq = np.mean(mquery_feature[mquery_index, :], axis=0)
        ap_tmp, CMC_tmp = evaluate(query_feature_new[i], query_label[i], query_cam[i], gallery_feature_new, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label)  # average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))
