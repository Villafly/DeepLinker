import numpy as np
import scipy.sparse as sp
import torch
import random
import copy
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,f1_score,roc_auc_score
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_cora_data(break_portion,path="./data/cora/", dataset="cora"):
    """Load Cora network, generate training, validation and test set for link prediction task"""
    print('Loading {} dataset...'.format(dataset))
    
    #load graph
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    ori_adj = copy.deepcopy(adj.todense())   
    features = np.array(features.todense())  
    
    #generate train,validation and test set 
    train_adj,idx_test_positive = break_links(adj.todense(),break_portion)
    idx_train = negative_sampling(train_adj,idx_test_positive) 
    idx_test = test_negative_sampling(ori_adj,idx_test_positive,idx_train)
    idx_train, idx_val = train_test_split(idx_train,test_size=0.05, random_state=1)  
    
      
    
    return torch.FloatTensor(ori_adj), torch.FloatTensor(train_adj), torch.FloatTensor(features), torch.LongTensor(idx_train), torch.LongTensor(idx_val), torch.LongTensor(idx_test)


def break_links(adj,break_portion):
    idx_test = []
    N = np.shape(adj)[0]
    cnt = 0
    break_num = math.ceil(break_portion*np.sum(adj)/2)    
    while cnt < int(break_num):
        x_cor = random.randint(0,N-1)
        y_cor = random.randint(0,N-1)
        if adj[x_cor,y_cor] == 1 and np.sum(adj[x_cor,:])!=1 and np.sum(adj[y_cor,:])!=1:
            idx_test.extend([x_cor*N + y_cor,y_cor*N + x_cor])
            adj[x_cor,y_cor] = adj[y_cor,x_cor] =0
            cnt += 1    
    return adj, idx_test
    
def negative_sampling(adj, idx_test):
    # one positive combined with one negative, sample list for train
    # highOrder_adj represent nodes which have no connection in high order
    idx_train_positive = np.array(list(np.where(np.array(adj).flatten() !=0))[0])
    train_positive_num = idx_train_positive.shape[0]    
    zero_location = list(np.where(np.array(adj).flatten() == 0))[0]
    temp = np.isin(zero_location, idx_test)    
    idx_train_negative = np.random.choice(zero_location[np.where(temp == False)], size = train_positive_num, replace=False)
    idx_train = np.hstack((idx_train_negative, idx_train_positive))
    np.random.shuffle(idx_train)
    print('train negative sampling done')
    return idx_train


def _test_negative_sampling(adj,idx_test_list_positive,idx_train_noTensor):
    
    N = np.shape(adj)[0]
    for i in range(len(idx_test_list_positive)):
            z = random.randint(0,N*N-1)
            while highOrder_adj[z] != 0 or z in idx_train_noTensor:
                z = random.randint(0,N*N-1)
            idx_test_list_positive.append(z)
    random.shuffle(idx_test_list_positive)
    idx_test_list = idx_test_list_positive 
    print 'test negative sampling done'
    return idx_test_list
    
def test_negative_sampling(adj,idx_test_positive,idx_train):
    idx_test_positive = np.array(idx_test_positive)
    test_positive_num = idx_test_positive.shape[0]
    adj = adj + np.eye(adj.shape[0])
    zero_location = list(np.where(np.array(adj).flatten() == 0))[0]
    choice_pos = np.isin(zero_location,idx_train)
    idx_test_negative = np.random.choice(zero_location[np.where(choice_pos == False)], size = test_positive_num, replace=False)
    idx_test = np.hstack((idx_test_positive, idx_test_negative ))
    np.random.shuffle(idx_test)
    return idx_test

def accuracy(output, labels):
    assert output.size() == labels.size()
    output = output > 0.5
    preds = output.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / float(len(labels))

def score_f1(output, labels):
    assert output.size() == labels.size()
    output = output > 0.5
    preds = output.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    label = labels.cpu().data.numpy()
    pred = preds.cpu().data.numpy()
    score = f1_score(label,pred,average='micro')
    return score
