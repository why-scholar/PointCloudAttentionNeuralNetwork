import sys
import os
# import psutil

import torch.nn as nn
import torch
import torch.nn.functional as F
# from layers import GraphAttentionLayer
from layers_tf import GraphAttentionLayer

# pid=os.getpid()
# p=psutil.Process(pid)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.outc=nclass

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.FC=nn.Parameter(torch.zeros(size=(nhid*nheads, self.outc)))
        nn.init.xavier_uniform_(self.FC.data, gain=1.414)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # def forward(self, x, adj):
    def forward(self, x):
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        # print('------x = torch.cat: ',  p.memory_info().rss/(1024*1024), ' MB')
        # print('------shape of x: ', x.shape)
        # x = self.out_att(x, adj)
        # return F.log_softmax(x, dim=1)

        x=x.view(x.shape[0],1,x.shape[1],x.shape[2])
        # print('------shape of x: ', x.shape)
        # print('------x=x.view: ',  p.memory_info().rss/(1024*1024), ' MB')

        x=F.max_pool2d(x,(x.shape[2],1))
        # print('------shape of x: ', x.shape)
        # print('------x=F.max_pool2d: ',  p.memory_info().rss/(1024*1024), ' MB')

        x=x.view(x.shape[0],x.shape[-1])
        # print('------shape of x: ', x.shape)
        # print('------x=x.view: ',  p.memory_info().rss/(1024*1024), ' MB')

        # print('shape of FC: ', self.FC.shape)
        x=torch.mm(x,self.FC)
        # print('------model: ',  p.memory_info().rss/(1024*1024), ' MB')

        return x