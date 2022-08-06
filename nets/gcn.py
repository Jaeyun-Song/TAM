"""
Code Reference: https://github.com/victorchen96/ReNode/blob/main/transductive/network/gcn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_geometric.nn import GCNConv


class StandGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(StandGCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
        # x = F.relu(self.conv1(x, edge_index))

        return x


class StandGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(StandGCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout_p = dropout

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, adj):
        x = self.conv1(x,adj)
        x = F.relu(x)
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, adj)

        return x


class StandGCNX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(StandGCNX, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x,p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def create_gcn(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = StandGCN1(nfeat, nhid, nclass, dropout,nlayer)
    elif nlayer == 2:
        model = StandGCN2(nfeat, nhid, nclass, dropout,nlayer)
    else:
        model = StandGCNX(nfeat, nhid, nclass, dropout,nlayer)
    return model
