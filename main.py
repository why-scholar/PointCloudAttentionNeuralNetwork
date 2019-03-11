import numpy as np
import argparse
import os
# import sys
import gc
# import viz
import time
from utils import accuracy
import random
import provider
# import psutil
import pynvml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from dataset import vkitti_dataset
# from gat.models import GAT
from models_tf import GAT

# pid=os.getpid()
# p=psutil.Process(pid)

# #initializing pynvml
# pynvml.nvmlInit()
# #get the handle of first GPU
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# print('before, memory of gpu used is ', meminfo.used/(1024*1024*8), ' MB')

parser = argparse.ArgumentParser(description='Point Cloud Attention Neural Network')

# Dataset
#parser.add_argument('--dataset', default='sema3d', help='Dataset name: sema3d|s3dis')
parser.add_argument('--cuda', action='store_true', default=False, help='Disables cuda training')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
# parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train.')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
BATCH_SIZE=args.batch_size
NUM_POINT=args.num_point
NUM_CLASSES=40 #modelnet40 dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#      torch.cuda.manual_seed(args.seed)

#ModelNet40 official train/test split
TRAIN_FILES=provider.getDataFiles(\
     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES=provider.getDataFiles(\
     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

model = GAT(nfeat=3, 
               nhid=args.hidden, 
               nclass=NUM_CLASSES, 
               dropout=args.dropout, 
               nheads=args.nb_heads, 
               alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                    lr=args.lr, 
                    weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()

# print('initlizing parameters and model: ',  p.memory_info().rss/(1024*1024), ' MB')

def train_one_epoch(epoch):
     print('Epoch: {:04d}'.format(epoch+1))

     is_training=True
     # t=time.time()
     model.train()

     #shuffle train files
     train_file_idxs=np.arange(0,len(TRAIN_FILES))
     np.random.shuffle(train_file_idxs)

     for fn in range(len(TRAIN_FILES)):
          print('---folder: ',str(fn))
          current_data,current_label=provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
          current_data=current_data[:,0:NUM_POINT,:]
          current_data,current_label,_=provider.shuffle_data(current_data,np.squeeze(current_label))
          # print('shape of current_label:', current_label.shape)
          # print('current_label:', current_label)
          # print('---load  current_data and current_label',  p.memory_info().rss/(1024*1024), ' MB')

          num_batches=current_data.shape[0]//BATCH_SIZE
          print('num_batches: ', num_batches)

          total_correct=0
          total_seen=0
          loss_sum=0

          for batch_idx in range(num_batches):
               print('------batch_idx: ', batch_idx)
               # print('------at the beginning of loop ',  p.memory_info().rss/(1024*1024), ' MB')
               # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
               # print('batch_idx, memory of gpu used is ', meminfo.used/(1024*1024*8), ' MB')

               optimizer.zero_grad()

               start_idx = batch_idx * BATCH_SIZE
               end_idx = (batch_idx+1) * BATCH_SIZE

               #augment batched point clouds by rotation and jittering
               rotated_data=provider.rotate_point_cloud(current_data[start_idx:end_idx,:,:])
               # print('------shape of rotated_data: ', rotated_data.shape)

               jittered_data=provider.jitter_point_cloud(rotated_data)
               # print('------shape of jittered_data: ', jittered_data.shape)
               # print('augment data ',  p.memory_info().rss/(1024*1024), ' MB')

               #showing current point cloud
               #pcd=open3d.PointCloud()
               #pcd.points=open3d.Vector3dVector(current_data[batch_idx,:,:])
               #open3d.draw_geometries([pcd])

               # adj=np.ones((jittered_data.shape[0],jittered_data.shape[1],jittered_data.shape[1]))
               # adj=torch.FloatTensor(adj)
               features=torch.FloatTensor(jittered_data)
               label=torch.LongTensor(current_label[start_idx:end_idx])

               # print('label:', label)

               # features,adj=Variable(features), Variable(adj)
               # print('------get adj and features ',  p.memory_info().rss/(1024*1024), ' MB')
               if args.cuda:
                    features = features.cuda()
               # output=model(features,adj)
               output=model(features)

               # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
               # print('model, memory of gpu used is ', meminfo.used/(1024*1024*8), ' MB')
               # print('------use model',  p.memory_info().rss/(1024*1024), ' MB')

               # print('type of output: ', type(output))
               # print('------shape of output: ', output.shape)
  
               #output: B*C, label:B
               #loss_train=F.nll_loss(output,label)
               loss_train=F.cross_entropy(output,label)
               # print('type of loss_train;', loss_train.requires_grad)
               _, acc_train=accuracy(output,label)
               # print('type of acc_train;', acc_train.requires_grad)

               # print('------before backward',  p.memory_info().rss/(1024*1024), ' MB')

               loss_train.backward()
               # print('------after backward',  p.memory_info().rss/(1024*1024), ' MB')

               total_correct+=acc_train.item()
               # print('type of total_correct;', type(total_correct))

               total_seen+=1
               loss_sum+=loss_train.item()
               # print('type of loss_sum;', type(loss_sum))

               # print('compute loss ',  p.memory_info().rss/(1024*1024), ' MB')

               del rotated_data
               del jittered_data
               # del adj
               del features
               del label
               del output
               del loss_train
               gc.collect()
               # print('------delete intermediate variable',  p.memory_info().rss/(1024*1024), ' MB')
               
               optimizer.step()
               # if batch_idx%BATCH_SIZE==0:
               #      print('batch_idx: ',batch_idx)
               #      optimizer.step()

          del current_data
          del current_label
          gc.collect()
          # print('---delete current_data and current_label',  p.memory_info().rss/(1024*1024), ' MB')

          print('---mean loss: {:.4f}'.format(loss_sum/float(num_batches)))
          print('---accuracy: {:.4f}'.format(total_correct/float(total_seen)))


def eval_one_epoch():
     model.eval()
     is_traing=False

     total_correct=0
     total_seen=0
     loss_sum=0

     total_seen_class=[0 for _ in range(NUM_CLASSES)]
     total_correct_class=[0 for _ in range(NUM_CLASSES)]

     for fn in range(len(TEST_FILES)):
          print('---',str(fn),'---')
          current_data,current_label=provider.loadDataFile(TEST_FILES[fn])
          current_data=current_data[:,0:NUM_POINT,:]
          current_label=np.squeeze(current_label)

          num_batches=current_data.shape[0] // BATCH_SIZE
          print('---num_batches: ', num_batches)

          for batch_idx in range(num_batches):
               start_idx = batch_idx * BATCH_SIZE
               end_idx = (batch_idx+1) * BATCH_SIZE

               data=current_data[start_idx:end_idx,:,:]
               # adj=np.ones((data.shape[0],data.shape[0]))
               # adj=torch.FloatTensor(adj)
               features=torch.FloatTensor(data)
               label=torch.LongTensor(current_label[start_idx:end_idx])

               if args.cuda:
                    features = features.cuda()
               # output=model(features,adj)
               output=model(features)
               
               #output: B*C, label:B
               loss_test=F.cross_entropy(output,label)
               correct, acc_test=accuracy(output,label)
               print('correct: ', correct)

               total_correct+=acc_test.item()
               total_seen+=1
               loss_sum+=loss_test.item()
               
               for i in range(start_idx, end_idx):
                    l=current_label[i].item()
                    total_seen_class[l]+=1
                    total_correct_class[l]+=(correct[i-start_idx]==l)

          print('eval mean loss: {:.4f}'.format(loss_sum/float(num_batches)))
          print('eval accuracy: {:.4f}'.format(total_correct/float(total_seen)))
          print('eval avg class accuracy: {:.4f}'.format(np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

#Train model
t_total=time.time()
# loss_values=[]
# bad_counter=0
# best=args.epochs+1
# best_epoch=0

for epoch in range(args.epochs):
     train_one_epoch(epoch)
     # eval_one_epoch()

     torch.save(model.state_dict(),'{}.pkl'.format(epoch))


print('Optimization Finished!')
print('Total time elapsed:', time.time()-t_total, 's')

#restore best model
print('loading ()th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
# eval_one_epoch()

