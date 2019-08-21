# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:15:04 2018

@author: wei
"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, LogisticRegression
import torch
from torch.autograd import Variable

class GAT(nn.Module):
    def __init__(self, K, node_num, nfeat, nhid,  sample_size, dropout, logistic_dim):
        super(GAT, self).__init__()
        self.gc1 = GraphConvolution(K, node_num, nfeat, nhid, sample_size[1], 'False','True')
        self.gc2 = GraphConvolution(1, node_num, K*nhid, logistic_dim, sample_size[0], 'False', 'False')
        self.gc3 = LogisticRegression(logistic_dim, 1)
        self.dropout = dropout

    def forward(self, nodes, node_neis, node_second_neis):
        node_nei_features = self.gc1(node_neis,node_second_neis)
        node_nei_features = F.dropout(node_nei_features, self.dropout, training=self.training)
        feature_num = node_nei_features.size()[1]
        features = range(feature_num)
        features_tensor = Variable(torch.LongTensor(features).cuda())
        node_features = torch.index_select(nodes,1,features_tensor)
        _node_features = self.gc2(node_features, node_nei_features)
        _node_features = F.dropout(_node_features, self.dropout, training=self.training)
        edge_features = self.gc3(_node_features)
        return edge_features
