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
    def __init__(self,K,node_num, nfeat, nhid, nclass, sampleSize, dropout):
        super(GAT, self).__init__()
        self.gc1 = GraphConvolution(K, node_num, nfeat, nhid, sampleSize[1],'False','True')
        self.gc2 = GraphConvolution(1, node_num, K*nhid, 14*nclass, sampleSize[0],'False','False')
        #self.gc3 = GraphConvolution(1, node_num, 4*7*nclass, 7*nclass, 'False','False')
        self.gc6 = LogisticRegression(14*nclass,1)
        self.dropout = dropout

    def forward(self, x, x_nei, x_nei2):
#        x = F.dropout(x, self.dropout, training=self.training)
        x_1 = self.gc1(x_nei,x_nei2)
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        nei_, feature_size = x_1.size()
        feature_size_list = range(feature_size)
        feature_size_tensor = Variable(torch.LongTensor(feature_size_list).cuda())
        x_features = torch.index_select(x,1,feature_size_tensor)
        x_2 = self.gc2(x_features, x_1)
        x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_6 = self.gc6(x_2)
        return x_6
