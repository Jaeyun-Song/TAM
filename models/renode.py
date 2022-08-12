"""
Code Reference: https://github.com/victorchen96/ReNode/blob/main/transductive/imb_loss.py
and https://github.com/victorchen96/ReNode/blob/main/transductive/load_data.py
"""


import numpy as np
# from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import random
import copy
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss,dim=1)

    return focal_loss

class IMB_LOSS:
    def __init__(self,loss_name,data,idx_info,factor_focal,factor_cb):
        self.loss_name = loss_name
        self.device    = device
        self.cls_num   = data.num_classes

        #train_size = [len(x) for x in data.train_node]
        train_size = [len(x) for x in idx_info]
        train_size_arr = np.array(train_size)
        train_size_mean = np.mean(train_size_arr)
        train_size_factor = train_size_mean / train_size_arr

        #alpha in re-weight
        self.factor_train = torch.from_numpy(train_size_factor).type(torch.FloatTensor)

        #gamma in focal
        self.factor_focal = factor_focal

        #beta in CB
        weights = torch.from_numpy(np.array([1.0 for _ in range(self.cls_num)])).float()

        if self.loss_name == 'focal':
            weights = self.factor_train

        if self.loss_name == 'cb-softmax':
            beta = factor_cb
            effective_num = 1.0 - np.power(beta, train_size_arr)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.cls_num
            weights = torch.tensor(weights).float()

        self.weights = weights.unsqueeze(0).to(device)



    def compute(self,pred,target):

        if self.loss_name == 'ce':
            return F.cross_entropy(pred,target,weight=None,reduction='none')

        elif self.loss_name == 're-weight':
            return F.cross_entropy(pred,target,weight=self.factor_train.to(self.device),reduction='none')

        elif self.loss_name == 'focal':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            return focal_loss(labels_one_hot,pred,weights,self.factor_focal)

        elif self.loss_name == 'cb-softmax':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            pred = pred.softmax(dim = 1)
            temp_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights,reduction='none') 
            return torch.mean(temp_loss,dim=1)

        else:
            raise Exception("No Implentation Loss")


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def index2dense(edge_index,nnode=2708):

    indx = edge_index.cpu().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj

def index2adj(inf,nnode = 2708):

    indx = inf.numpy()
    print(nnode)
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    return adj

def adj2index(inf):

    where_new = np.where(inf>0)
    new_edge = [where_new[0],where_new[1]]
    new_edge_tensor = torch.from_numpy(np.array(new_edge))
    return new_edge_tensor

def log_opt(opt,log_writer):
    for arg in vars(opt): log_writer.write("{}:{}\n".format(arg,getattr(opt,arg)))

def to_inverse(in_list,t=1):

    in_arr = np.array(in_list)
    in_mean = np.mean(in_arr)
    out_arr = in_mean / in_arr
    out_arr = np.power(out_arr,t)

    return out_arr


def get_renode_weight(data, data_train_mask,base_weight,max_weight):

    ##hyperparams##
    rn_base_weight = base_weight
    rn_scale_weight = max_weight - base_weight
    assert rn_scale_weight in [0.5 , 0.75, 1.0, 1.25, 1.5]

    ppr_matrix = data.Pi  #personlized pagerank
    gpr_matrix = torch.tensor(data.gpr).float() #class-accumulated personlized pagerank

    base_w  = rn_base_weight
    scale_w = rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data_train_mask.int().ne(1)#unlabled node

    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix

    label_matrix = F.one_hot(data.y,gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0
    rn_matrix = torch.mm(ppr_matrix,gpr_rn).to(label_matrix.device)
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node

    #computing the ReNode Weight
    train_size    = torch.sum(data_train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight.to(data_train_mask.device)
    rn_weight = rn_weight * data_train_mask.float()

    return rn_weight








