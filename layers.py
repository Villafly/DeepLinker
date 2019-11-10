# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:27:02 2018

@author: wei
"""

import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):  
        super(LogisticRegression, self).__init__()  
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden) # hidden layer  
        self.active = nn.Sigmoid() ####  # output layer  
    def forward(self,x):
        node_num, out_features = x.size()
        even_tensor = Variable(torch.LongTensor(range(node_num)[1::2]).cuda())
        odd_tensor = Variable(torch.LongTensor(range(node_num)[::2]).cuda())
        properity = torch.mul(torch.index_select(x,0,odd_tensor),torch.index_select(x,0,even_tensor))
        value = self.parameter(properity)
        out = self.active(value)
        return out.squeeze()
        

class GraphConvolution(Module):

    def __init__(self,K,node_num, in_features, out_features, sample_size,bias='False',cat ='True',trainAttention='True'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sample_size = sample_size
        self.node_num = node_num
        self.K = K 
        self.trainAttention = 1
        self.leakyRelu = nn.LeakyReLU(0.2) 
        self.cat = cat
        self.weight = Parameter(torch.Tensor(K,in_features, out_features))
        self.a = Parameter(torch.Tensor(K, 2*self.out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def cal_alpha_matrix(self,k, head_support, tail_support):
        repeated = head_support.repeat(1,self.sample_size).view(-1,self.out_features)
        tiled = tail_support
        alpha_matrix = torch.cat((repeated, tiled),1)
        dense = torch.matmul(alpha_matrix,self.a[k]) 
        dense = self.leakyRelu(torch.matmul(alpha_matrix,self.a[k]))
        alpha = F.softmax(dense.view(-1,self.sample_size), dim = 1)
        dense = alpha.view(-1,1)
        combination_slices =  dense.repeat(1,self.out_features)
        return combination_slices
    
    def cal_node_feature(self,adj_alpha,tail_support):
        unsum_feature = torch.mul(adj_alpha,tail_support)
        features_3dim = unsum_feature.view(-1,self.sample_size,self.out_features)
        trans_features = torch.transpose(features_3dim,1,2)
        sum_features = torch.sum(trans_features, 2, True)
#        features = torch.transpose(sum_features, 1,2).squeeze()
        return sum_features.squeeze()

    def forward(self, input_head, input_tail):
        # in the first layer,transfer neighborMatrix to list
        for k in range(self.K):
            head_support = torch.mm(input_head, self.weight[k])
            tail_support = torch.mm(input_tail, self.weight[k])
            alpha_matrix = self.cal_alpha_matrix(k,head_support,tail_support)
            if trainAttention == 'False':
                x,y =  alpha_matrix.size()
                alpha_matrix = Variable((torch.ones(x,y)).cuda())            
            node_features = self.cal_node_feature(alpha_matrix,tail_support)
            if self.cat == 'True': 
                 _x = F.relu(node_features)
                 if k == 0:
        				temp_out = _x
                 else:
        				output = torch.cat((temp_out, _x),1); temp_out = output;
            
            else:
                output = node_features
                return output
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
