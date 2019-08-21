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
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import load_cora_data,accuracy,score_f1
from models import GAT
from sample_neighbors import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
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
parser.add_argument('--batch_size', type = int, default=64,
                    help='set the trainning batch size')
parser.add_argument('--test_batch_size', type = int, default=64,
                    help='set the test batch size for test dataset.')
parser.add_argument('--sample_size', type = str, default= '5,5',
                    help='set the sample size.')
parser.add_argument('--break_portion', type = float, default= 0.1,
                    help='set the break portion size.')
parser.add_argument('--patience', type = int, default=100, help='Patience')
parser.add_argument('--logistic_dim', type = int, default=28,
                    help='set the logistic parameter size ')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
ori_adj, train_adj, features,idx_train, idx_val, idx_test = load_cora_data(args.break_portion)

#'test'
node_num = features.numpy().shape[0]
sample_size = map(int, args.sample_size.split(','))
model = GAT(K = 8, node_num = features.numpy().shape[0],nfeat=features.numpy().shape[1],nhid=args.hidden, sample_size = sample_size, dropout=args.dropout,logistic_dim=args.logistic_dim)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.BCELoss() 

if torch.cuda.device_count() > 1:
    print("Let's use multi-GPUs!")
    model = nn.DataParallel(model)
else:
    print("Only use one GPU")

labels = train_adj.view(node_num*node_num).type_as(idx_train).type(torch.FloatTensor)
ori_labels = ori_adj.view(node_num*node_num).type(torch.FloatTensor)
train_adj,ori_labels = Variable(train_adj), Variable(ori_labels) 
features,labels = Variable(features), Variable(labels)

if args.cuda:
    model = model.cuda()
    features = features.cuda()
    train_adj = train_adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    criterion = criterion.cuda()
    ori_labels = ori_labels.cuda()

writer = SummaryWriter(comment = 'Cora_hidden_{}_lr_{}_batch_{}_SampleSize_{}_test_batch_size_{}'.format(args.hidden , args.lr , args.batch_size , args.sample_size,args.test_batch_size))

train_batches = get_neighbors(idx_train, args.sample_size,labels,train_adj, args.batch_size)
val_batches   = get_neighbors(idx_val, args.sample_size, labels, train_adj, len(idx_val))
test_batches = get_neighbors(idx_test, args.sample_size,ori_labels,train_adj, args.test_batch_size)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    batch_num = len(train_batches); loss_train_sum= 0; accu_train_sum = 0;loss_val_sum= 0; accu_val_sum = 0
    _idx_val, _idx_neighbor_val, _idx_neighbor2_val, _targets_val = val_batches[0]
    for train_data in train_batches:
        _idx_train, _idx_neighbor_train, _idx_neighbor2_train, _targets_train = train_data
        output = model(torch.index_select(features,0,_idx_train),torch.index_select(features,0,_idx_neighbor_train),torch.index_select(features,0,_idx_neighbor2_train))
        loss_train = criterion(output, _targets_train.type(torch.cuda.FloatTensor))
        acc_train = accuracy(output, _targets_train.type(torch.cuda.FloatTensor))
        model.train()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        loss_train_sum += loss_train.data[0]; accu_train_sum += acc_train.data[0]

    if not args.fastmode:
        model.eval()
        output = model(torch.index_select(features,0,_idx_val),torch.index_select(features,0,_idx_neighbor_val),torch.index_select(features,0,_idx_neighbor2_val))
        loss_val = criterion(output, _targets_val.type(torch.cuda.FloatTensor))
        acc_val = accuracy(output, _targets_val.type(torch.cuda.FloatTensor))
        loss_val_sum += loss_val.data[0];accu_val_sum += acc_val.data[0]

    if (epoch +1)%1 ==0:
        test(epoch,test_batches)
    loss_train = loss_train_sum/float(batch_num)
    accu_train = accu_train_sum/float(batch_num)
    loss_val = loss_val.data[0]
    accu_val = acc_val.data[0]
    print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(accu_train),
              'loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(accu_val),
              'time: {:.4f}s'.format(time.time() - t))
    writer.add_scalars('loss',{'train_loss':loss_train,'val_loss':loss_val},epoch)
    writer.add_scalars('acc',{'train_acc':accu_train , 'val_acc':accu_val},epoch)
    
    return epoch+1, loss_train, loss_val, accu_train, acc_val


def test(epoch,test_batches):
    model.eval()
    batch_num = len(test_batches); loss_test_sum= 0; accu_test_sum = 0;f1_score_sum= 0; 
    for test_data in test_batches:
        _idx_test, _idx_neighbor_test,_idx_neighbor2_test,_targets_test  = test_data
        output = model(torch.index_select(features,0,_idx_test),torch.index_select(features,0,_idx_neighbor_test),torch.index_select(features,0,_idx_neighbor2_test))
        #loss_test = criterion(output, torch.index_select(ori_labels,0,targets))
        loss_test = criterion(output, _targets_test.type(torch.cuda.FloatTensor))
        
        accu_test = accuracy(output,_targets_test.type(torch.cuda.FloatTensor))
        f1_score = score_f1(output,_targets_test.type(torch.cuda.FloatTensor))
        loss_test_sum += loss_test.data[0]
        accu_test_sum += accu_test.data[0]
        f1_score_sum += f1_score      

    loss_test = loss_test_sum/float(batch_num)
    accu_test = accu_test_sum/float(batch_num)
    f1_score = f1_score_sum/float(batch_num)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(accu_test),
          "f1_score = {:.4f}".format(f1_score))

    writer.add_scalar('loss_test' , loss_test,epoch)
    writer.add_scalar('acc_test' , accu_test,epoch)
    writer.add_scalar('f1_score' , f1_score,epoch)


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    epoch, loss_train, loss_val, acc_train, acc_val = train(epoch)
    torch.save(model.state_dict(), './patience/{}.pkl'.format(epoch))
    loss_values.append(loss_val)
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('./patience/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[2][:-4])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./patience/*.pkl')
for file in files:
    epoch_nb = int(file.split('/')[2][:-4])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./patience/{}.pkl'.format(best_epoch)))
model_dict = model.state_dict()

test(epoch,test_batches)

#export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

