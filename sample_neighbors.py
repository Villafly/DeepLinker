# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:28:42 2018

@author: wei
"""
import numpy as np
import random
import numpy.core.numeric as _nx
import time
import threading
from multiprocessing import Pool

import torch
from torch.autograd import Variable

def edge2pair(idx, node_num):
    """convert idx to node pair index"""
    node_indexs = []
    for i in idx:
        node_indexs.extend(divmod(i, node_num))
    return node_indexs

class SampleThread(threading.Thread):
    def __init__(self,func,args=()):
        super(SampleThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
            
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
    return sub_arys
    
def to_numpy(x):
    if isinstance(x,Variable):
        return to_numpy(x.data)
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def batch_to_torch(*args):
    return_values = []
    for v in args:
        m = Variable(torch.LongTensor(v).cuda())
        if len(m.size()) != 1:
            m = m.view(-1,1).squeeze()
        return_values.append(m)
    return return_values 
              
def sample_neighbor(nodes, adj, sample_size):
    temp_row_idxs, temp_col_idxs  = np.where(adj != 0)
    row_idxs, row_idxs_cnt = np.unique(temp_row_idxs, return_counts=True)
    previous_cnt = 0
    sample_matrix = np.zeros((adj.shape[0], sample_size))
    for row, cnt in zip(row_idxs, row_idxs_cnt):
        if row not in nodes:
            previous_cnt += cnt
            continue
        tt = sample_size // cnt + 1
        node_sample = temp_col_idxs[previous_cnt:previous_cnt+cnt].tolist()
        random.shuffle(node_sample)
        node_sample = (node_sample*tt)[:sample_size]
        previous_cnt += cnt
        sample_matrix[row, :] = node_sample
    return sample_matrix[nodes].astype(np.int)
    
def multi_sample(node_list,adj,sample_size):
    pool = Pool()
    multi_results = []
    for ids in np.array_split(node_list,10):
        multi_results.append(pool.apply_async(sample_neighbor,args=(ids, adj, sample_size)))
    pool.close()
    pool.join()  
    multi_results = [item.get() for item in multi_results]
    node_matrix = reduce(lambda x,y: np.vstack((x,y)), multi_results)

    return node_matrix
    
def get_neighbors(idx,sample_size,labels,adj, batch_size= 256):
    #pre for batch, sample data,
    sample_size = map(int, sample_size.split(','))
    idx = to_numpy(idx)
    adj = to_numpy(adj)
    node_num = np.shape(adj)[0]
    node_list = edge2pair(idx,node_num)
    
    #sample neighbors
    node_neighbor_matrix = multi_sample(node_list,adj,sample_size[0])  
    node_nei_list = node_neighbor_matrix.flatten().tolist()    
    node_neighbor_2_matrix = multi_sample(node_nei_list,adj,sample_size[1]) 
    print("sample second-order neighbors done.")
    nodes_neighbors = []
    for chunk_id, ids in enumerate(_array_split(idx, batch_size)):
        targets = labels[ids]
        node_list_batch = edge2pair(ids, node_num)
        node_batch_num = len(node_list_batch)
        node_neighbor_matrix_batch = node_neighbor_matrix[chunk_id*node_batch_num : (chunk_id + 1)*node_batch_num, :]
        node_neighbor_2_matrix_batch = node_neighbor_2_matrix[chunk_id*node_batch_num*sample_size[0] : (chunk_id + 1)*node_batch_num*sample_size[0], :]
        _node_list,_node_neighbor_matrix,_node_neighbor_2_matrix = batch_to_torch(np.array(node_list_batch),node_neighbor_matrix_batch, node_neighbor_2_matrix_batch)
        node_neighbor = [_node_list, _node_neighbor_matrix, _node_neighbor_2_matrix, targets]
        nodes_neighbors.append(node_neighbor)
       
    return nodes_neighbors

