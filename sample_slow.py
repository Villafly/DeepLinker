# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:28:42 2018

@author: wei
"""
from functools import partial
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import random
from random import shuffle
import math
import numpy.core.numeric as _nx
import threading
from multiprocessing import Pool
def to_numpy(x):
    if isinstance(x,Variable):
        return to_numpy(x.data)
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def __batch_to_torch(*args):
    return_list = []
    for cnt,v in enumerate(args):
        m = Variable(torch.LongTensor(v).cuda())
        if len(m.size()) != 1:
            m = m.view(-1,1).squeeze()
        return_list.append(m)
    return return_list

def edge2pair(idx, node_num):
# change idx to node_pair then return list with first and second elements represent head tail
    node_list = []
    for i in idx:
        node_list.extend(divmod(i, node_num))
    return node_list

def sample_neighbor(node_list, adj, sample_size):
    temp_row_idxs, temp_col_idxs  = np.where(adj != 0)
    row_idxs, row_idxs_cnt = np.unique(temp_row_idxs, return_counts=True)
    previous_cnt = 0
    sample_matrix = np.zeros((adj.shape[0], sample_size))
    for row, cnt in zip(row_idxs, row_idxs_cnt):
        if row not in node_list:
            previous_cnt += cnt
            continue
        tt = sample_size // cnt + 1
        node_sample = temp_col_idxs[previous_cnt:previous_cnt+cnt].tolist()
        random.shuffle(node_sample)
        node_sample = (node_sample*tt)[:sample_size]
        previous_cnt += cnt
        sample_matrix[row, :] = node_sample
    return sample_matrix[node_list].astype(np.int)

def sample_test(idx,sample_size,adj,labels):
    labels, idx = to_numpy(labels).astype(int), to_numpy(idx)
    targets = idx
    sample_size = map(int, sample_size.split(','))
    adj = to_numpy(adj)
    node_num = np.shape(adj)[0]
    node_list = edge2pair(idx,node_num)
    train_num = len(node_list)
    node_neighbor_matrix = sample_neighbor(node_list, adj,sample_size[0])
    node_nei_list = node_neighbor_matrix.flatten().tolist()
    node_neighbor_2_matrix = sample_neighbor(node_nei_list,adj,sample_size[1])
    node_list,node_neighbor_matrix,node_neighbor_2_matrix, targets = __batch_to_torch(np.array(node_list),node_neighbor_matrix,node_neighbor_2_matrix,targets)
    return node_list, node_neighbor_matrix,node_neighbor_2_matrix,targets

def _array_split(idx,batch_size,axis=0):
    N = len(idx)+1
    div_points = [0]
    for i in range(1,N):
        if divmod(i,batch_size)[1] == 0:
            div_points.append(i)
    sub_arys=[]
    sary = _nx.swapaxes(idx, axis, 0)
    for i in range(len(div_points)-1):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
    sub_arys.append(_nx.swapaxes(sary[-batch_size:], axis, 0))
    return sub_arys

def sample_test_batch(idx,sample_size,adj,batch_size = 256):
    idx =  to_numpy(idx)
    sample_size = map(int, sample_size.split(','))
    adj = to_numpy(adj)
    node_num = np.shape(adj)[0]
    return_list = []
    for chunk_id, ids in enumerate(_array_split(idx, batch_size)):
        targets = ids
        node_list = edge2pair(ids,node_num)
        train_num = len(node_list)
        node_neighbor_matrix = sample_neighbor(node_list, adj,sample_size[0])
        node_nei_list = node_neighbor_matrix.flatten().tolist()
        node_neighbor_2_matrix = sample_neighbor(node_nei_list,adj,sample_size[1])
        node_list,node_neighbor_matrix,node_neighbor_2_matrix ,targets = __batch_to_torch(np.array(node_list),node_neighbor_matrix,node_neighbor_2_matrix,targets)
        son_list = [node_list, node_neighbor_matrix,node_neighbor_2_matrix,targets]
        return_list.append(son_list)
    return return_list

def iterate_return(idx,sample_size,labels,adj, batch_size= 256):
    #pre for batch, sample data,
    sample_size = map(int, sample_size.split(','))
    labels, idx = to_numpy(labels).astype(int), to_numpy(idx)
    adj = to_numpy(adj)
    node_num = np.shape(adj)[0]
    return_list = []
    for chunk_id, ids in enumerate(_array_split(idx, batch_size)):
        targets = labels[ids]
        node_list = edge2pair(ids,node_num)
        train_num = len(node_list)
        node_neighbor_matrix = sample_neighbor(node_list, adj,sample_size[0])
        node_nei_list = node_neighbor_matrix.flatten().tolist()
        node_neighbor_2_matrix = sample_neighbor(node_nei_list,adj,sample_size[1])
        node_list,node_neighbor_matrix,node_neighbor_2_matrix, targets = __batch_to_torch(np.array(node_list),node_neighbor_matrix,node_neighbor_2_matrix,targets)
        son_return = []
        for i in  node_list, node_neighbor_matrix, node_neighbor_2_matrix, targets:
            son_return.append(i)
        return_list.append(son_return)
    return return_list
