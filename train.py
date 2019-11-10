#  -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:43:54 2018

@author: wei
"""

from __future__ import division
from __future__ import print_function
import torch
import time
import os
import glob
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from models import GAT
import cPickle as pickle
#from load_karate import load_karate
from sample import *
from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default= 5e-4,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--K', type=int, default=8,
                    help='Number of attention.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batchSize', type = int, default=32,
                    help='set the batchSize')
parser.add_argument('--testBatchSize', type = int, default=32,
                    help='set the batchSize for test dataset.')
parser.add_argument('--sampleSize', type = str, default= '20,20',
                    help='set the sampleSize.')
parser.add_argument('--breakPortion', type = float, default= 0.1,
                    help='set the break portion.')
parser.add_argument('--patience', type = int, default=50, help='Patience')
parser.add_argument('--trainAttention', type = int, default= 1,
                    help='Train attention weight or not')



args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
datasetName = 'cora'
ori_adj, adj, features,idx_train, idx_val, idx_test = load_cora_data(args.breakPortion)
#ori_adj, adj, features, labels, idx_train, idx_val, idx_test = load_pubmed_data(args.breakPor)


node_num = features.numpy().shape[0]
sampleSize = map(int, args.sampleSize.split(','))
model = GAT(K = 8, node_num = features.numpy().shape[0],nfeat=features.numpy().shape[1],nhid=args.hidden,nclass=2,sampleSize = sampleSize, dropout=args.dropout,trainAttention=args.trainAttention)


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.device_count() > 1:
    print("Let's use multi-GPUs!")
    model = nn.DataParallel(model)
else:
    print("Only use one GPU")

labels = adj.view(node_num*node_num).type_as(idx_train)
labels = labels.type(torch.FloatTensor)
ori_labels = ori_adj.view(node_num*node_num).type_as(idx_train)
ori_labels = ori_labels.type(torch.FloatTensor)
criterion = nn.BCELoss() 
adj,ori_labels = Variable(adj), Variable(ori_labels) 
features = Variable(features)
labels = Variable(labels)

if args.cuda:
    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    criterion = criterion.cuda()
    ori_labels = ori_labels.cuda()

writer = SummaryWriter(comment = 'ori_0.1_hidden_{}_lr_{}_#batch_{}_SampleSize_{}_testBatchSize_{}'.format(args.hidden , args.lr , args.batchSize , args.sampleSize,args.testBatchSize))


train_batches = iterate_return(idx_train, args.sampleSize,labels,adj, args.batchSize)
val_batches   = iterate_return(idx_val, args.sampleSize, labels, adj, args.batchSize )
test_batches  = sample_test_batch(idx_test, args.sampleSize,adj, args.testBatchSize)
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    batch = 0; loss_train_sum= 0; accu_train_sum = 0;loss_val_sum= 0; accu_val_sum = 0
    _idx_val, _idx_neighbor_val, _idx_neighbor2_val, _targets_val = val_batches[0]
    for epoch_cnt,i in enumerate(train_batches):
        _idx_train, _idx_neighbor_train, _idx_neighbor2_train, _targets_train = i
        output = model(torch.index_select(features,0,_idx_train),torch.index_select(features,0,_idx_neighbor_train),torch.index_select(features,0,_idx_neighbor2_train))      
        loss_train = criterion(output, _targets_train.type(torch.cuda.FloatTensor))
        acc_train = accuracy(output, _targets_train.type(torch.cuda.FloatTensor))
        model.train()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        loss_train_sum += loss_train.data[0]; accu_train_sum += acc_train.data[0]
        batch += 1

    if not args.fastmode:
        model.eval()
        output = model(torch.index_select(features,0,_idx_val),torch.index_select(features,0,_idx_neighbor_val),torch.index_select(features,0,_idx_neighbor2_val))
        loss_val = criterion(output, _targets_val.type(torch.cuda.FloatTensor))
        acc_val = accuracy(output, _targets_val.type(torch.cuda.FloatTensor))
        loss_val_sum += loss_val.data[0];accu_val_sum += acc_val.data[0]

    if (epoch +1)%10 ==0:
        test(epoch,test_batches)

    print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train_sum/float(batch)),
              'acc_train: {:.4f}'.format(accu_train_sum/float(batch)),
              'loss_val: {:.4f}'.format(loss_val.data[0]),
              'acc_val: {:.4f}'.format(acc_val.data[0]),
              'time: {:.4f}s'.format(time.time() - t))
    writer.add_scalars('loss',{'train_loss':loss_train_sum / float(batch),'val_loss':loss_val_sum/float(batch)},epoch)
    writer.add_scalars('acc',{'train_acc':accu_train_sum / float(batch) , 'val_acc':accu_val_sum / float(batch)},epoch)

    if args.cuda == True:
        return epoch+1, loss_train_sum/float(batch), loss_val.data[0], accu_train_sum/float(batch), acc_val.data[0]
    else:
        return epoch+1, loss_train.data.numpy(), loss_val.data.numpy(), acc_train.data.numpy(), acc_val.data.numpy()

def test(epoch,test_batches):
    model.eval()
    batch = 0; loss_test_sum= 0; accu_test_sum = 0;f1_score_sum= 0; auc_sum = 0
    for i in test_batches:
        _idx_test, _idx_neighbor_test,_idx_neighbor2_test,targets  = i
        output = model(torch.index_select(features,0,_idx_test),torch.index_select(features,0,_idx_neighbor_test),torch.index_select(features,0,_idx_neighbor2_test))        
        loss_test = criterion(output, torch.index_select(ori_labels,0,targets))
        acc_test = accuracy(output,torch.index_select(ori_labels,0,targets).type(torch.cuda.FloatTensor))
        f1_score = score_f1(output,torch.index_select(ori_labels,0,targets).type(torch.cuda.FloatTensor))
        try:
            auc = roc_auc_score(torch.index_select(ori_labels,0,targets).cpu().data.numpy(),output.cpu().data.numpy())
        except:
            auc = 1

        loss_test_sum += loss_test.data[0]
        accu_test_sum += acc_test.data[0]
        f1_score_sum += f1_score
        auc_sum += auc
        batch += 1

    print("Test set results:",
          "loss= {:.4f}".format(loss_test_sum/float(batch)),
          "accuracy= {:.4f}".format(accu_test_sum/float(batch)),
          "f1_score = {:.4f}".format(f1_score_sum/float(batch)),
          "auc = {:.4f}".format(auc_sum/float(batch)))

    writer.add_scalar('loss_test' , loss_test_sum/float(batch),epoch)
    writer.add_scalar('acc_test' , accu_test_sum/float(batch),epoch)
    writer.add_scalar('f1_score' , f1_score_sum/float(batch),epoch)
    writer.add_scalar('auc' , auc_sum/float(batch),epoch)

def link_prediction():
    model.eval()
    output = model(features, adj)
    accu = link_prediction_accuracy_2label(ori_adj, adj, output)
    print ('link prediction accuracy is {}'.format(accu))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    epoch, loss_train, loss_val, acc_train, acc_val = train(epoch)
    loss_values.append(loss_val)
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    torch.save(model, '{}.modelPkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('./Patience/*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./Patience/*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
model_dict = model.state_dict()
test(epoch,test_batches)

#export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

