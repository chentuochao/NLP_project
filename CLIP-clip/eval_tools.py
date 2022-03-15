import numpy as np
import numpy.linalg as LA
from numpy import argpartition as argpart
import torch

def cosine_sims(qfeats, ifeats):
    # qfeats: features of queried text
    # ifeats: features of images in dataset
    nrm_qfeats = qfeats / LA.norm(qfeats, axis=1, keepdims=True)
    nrm_ifeats = ifeats / LA.norm(ifeats, axis=1, keepdims=True)
    t_nrm_qfeats = torch.Tensor(nrm_qfeats)
    t_nrm_ifeats = torch.Tensor(nrm_ifeats)
    t_sim_maps = torch.matmul(t_nrm_qfeats, t_nrm_ifeats.T)
    sim_maps = t_sim_maps.cpu().numpy()
    
    return sim_maps

def pick_topK(sim_map, K):
    # pick the indexes that has the top-K similarity
    topK_inds = argpart(sim_map, -K, axis=1)[:, -K:]

    return topK_inds

def eval_gender_biases(topK_ids, gdr_attrs, gdr_dict):
    num_query = topK_ids.shape[0]
    gdr_biases = np.zeros(num_query)
    for idx, topK_id in enumerate(topK_ids):
        gdr_labels = gdr_attrs[topK_id]
        num_male = sum([x == gdr_dict['male'] for x in gdr_labels])
        num_female = sum([x == gdr_dict['female'] for x in gdr_labels])
        bias = (num_male - num_female) / max((num_male + num_female), 1e-5)
        gdr_biases[idx] = bias
    
    # stats_bias = dict(mean=None, std=None)
    # stats_bias['mean'] = np.mean(gdr_biases)
    # stats_bias['std'] = np.std(gdr_biases)

    return np.mean(gdr_biases)

def eval_recalls(topK_ids, gt_ids):
    num_query = topK_ids.shape[0]
    results = np.zeros(num_query)
    for idx, topK_id in enumerate(topK_ids):
        if int(gt_ids[idx]) in set(topK_id):
            results[idx] = 1

    mean_recall = np.mean(results)

    return mean_recall
    
