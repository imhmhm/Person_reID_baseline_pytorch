#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(gen_g_dist, gen_gen_dist, q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V_high = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    original_gen_dist = np.concatenate(
      [np.concatenate([gen_gen_dist, gen_g_dist], axis=1),
       np.concatenate([gen_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_gen_dist = 2. - 2 * original_gen_dist   #np.power(original_dist, 2).astype(np.float32)
    original_gen_dist = np.transpose(1. * original_gen_dist/np.max(original_gen_dist,axis = 0))
    V_low = np.zeros_like(original_gen_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_gen_rank = np.argpartition( original_gen_dist, range(1,k1+1) )


    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_gen_index = k_reciprocal_neigh( initial_gen_rank, i, k1)

        # real query
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # gen query
        k_reciprocal_expansion_gen_index = k_reciprocal_gen_index
        for j in range(len(k_reciprocal_gen_index)):
            candidate = k_reciprocal_gen_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_gen_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_gen_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_gen_index = np.append(k_reciprocal_expansion_gen_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_gen_index = np.unique(k_reciprocal_expansion_gen_index)

        # high and low sets
        # k_reciprocal_index_high = np.intersect1d(k_reciprocal_expansion_index, k_reciprocal_expansion_gen_index)
        # k_reciprocal_index_low = np.union1d(k_reciprocal_expansion_index, k_reciprocal_expansion_gen_index)

        k_reciprocal_index_high = k_reciprocal_expansion_index
        k_reciprocal_index_low = k_reciprocal_expansion_gen_index

        weight = np.exp(-original_dist[i,k_reciprocal_index_high])
        V_high[i,k_reciprocal_index_high] = 1.*weight/np.sum(weight)

        weight = np.exp(-original_dist[i,k_reciprocal_index_low])
        V_low[i,k_reciprocal_index_low] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]


    if k2 != 1:

        # initial_rank_high = np.intersect1d(initial_rank[i,:k2], initial_gen_rank[i,:k2])
        # initial_rank_low = np.union1d(initial_rank[i,:k2], initial_gen_rank[i,:k2])

        initial_rank_high = initial_rank[i,:k2]
        initial_rank_low = initial_gen_rank[i,:k2]

        V_qe_high = np.zeros_like(V_high,dtype=np.float32)
        for i in range(all_num):
            V_qe_high[i,:] = np.mean(V_high[initial_rank_high,:],axis=0)
        V_high = V_qe_high
        del V_qe_high

        V_qe_low = np.zeros_like(V_low,dtype=np.float32)
        for i in range(all_num):
            V_qe_low[i,:] = np.mean(V_low[initial_rank_low,:],axis=0)
        V_low = V_qe_low
        del V_qe_low

    # del initial_rank
    del initial_rank_high
    del initial_rank_low
    del initial_rank
    del initial_gen_rank

    invIndex_high = []
    for i in range(all_num):
        invIndex_high.append(np.where(V_high[:,i] != 0)[0])
    invIndex_low = []
    for i in range(all_num):
        invIndex_low.append(np.where(V_low[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min_high = np.zeros(shape=[1,all_num],dtype=np.float32)
        temp_min_low = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero_high = np.where(V_high[i,:] != 0)[0]
        indNonZero_low = np.where(V_low[i,:] != 0)[0]
        indImages_high = []
        indImages_high = [invIndex_high[ind] for ind in indNonZero_high]
        indImages_low = []
        indImages_low = [invIndex_low[ind] for ind in indNonZero_low]
        for j in range(len(indNonZero_high)):
            temp_min_high[0,indImages_high[j]] = temp_min_high[0,indImages_high[j]]+ np.minimum(V_high[i,indNonZero_high[j]],V_high[indImages_high[j],indNonZero_high[j]])
        for j in range(len(indNonZero_low)):
            temp_min_low[0,indImages_low[j]] = temp_min_low[0,indImages_low[j]]+ np.minimum(V_low[i,indImages_low[j]],V_low[indImages_low[j],indImages_low[j]])

        jaccard_dist[i] = 1- (1 * temp_min_high/(2.-temp_min_high) + 0 * temp_min_low/(2.-temp_min_low))

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V_high
    del V_low
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
