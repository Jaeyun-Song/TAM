import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


## Jensen-Shanon Divergence ##
def compute_jsd(dist1, dist2):
    dist_mean = (dist1 + dist2) / 2.
    jsd = (F.kl_div(dist_mean.log(), dist1, reduction = 'none') + F.kl_div(dist_mean.log(), dist2, reduction = 'none')) / 2.
    return jsd


## TAM ##
@torch.no_grad()
def compute_tam(output, edge_index, label, train_mask, aggregator, class_num_list=None, temp_phi = None, temp_gamma = None):
    n_cls = label.max().item() + 1

    # Apply class-wise temperature
    cls_num_list = torch.FloatTensor(class_num_list).to(output.device)
    cls_num_ratio = cls_num_list / cls_num_list.sum()
    cls_num_ratio = cls_num_ratio * temp_gamma + (1- temp_gamma)
    max_beta = torch.max(cls_num_ratio)
    cls_temperature = (temp_phi * (cls_num_ratio + 1 - max_beta)).unsqueeze(0)
    temp = 1 / cls_temperature

    # Predict unlabeled nodes
    agg_out = F.softmax(output.clone().detach()/temp, dim=1)
    agg_out[train_mask] = F.one_hot(label[train_mask].clone(), num_classes=n_cls).float() # only use labeled nodes
    neighbor_dist = aggregator(agg_out, edge_index)[train_mask] # (# of labeled nodes, # of classes)

    # Compute class-wise connectivity matrix
    connectivity_matrix= []
    for c in range(n_cls):
        c_mask = (label[train_mask] == c)
        connectivity_matrix.append(neighbor_dist[c_mask].mean(dim=0))
    connectivity_matrix= torch.stack(connectivity_matrix, dim=0)

    # Preprocess class-wise connectivity matrix and NLD for numerical stability
    center_mask = F.one_hot(label[train_mask].clone(), num_classes=n_cls).bool()
    neighbor_dist[neighbor_dist<1e-6] = 1e-6
    connectivity_matrix[connectivity_matrix<1e-6] = 1e-6

    # Compute ACM
    acm = (neighbor_dist[center_mask].unsqueeze(dim=1) / torch.diagonal(connectivity_matrix).unsqueeze(dim=1)[label[train_mask]]) \
                * (connectivity_matrix[label[train_mask]] / neighbor_dist)
    acm[acm>1] = 1
    acm[center_mask] = 1

    # Compute ADM
    cls_pair_jsd = compute_jsd(connectivity_matrix.unsqueeze(dim=0), connectivity_matrix.unsqueeze(dim=1)).sum(dim=-1) # distance between classes
    cls_pair_jsd[cls_pair_jsd<1e-6] = 1e-6
    self_kl = compute_jsd(neighbor_dist, connectivity_matrix[label[train_mask]]).sum(dim=-1,keepdim=True) # devation from self-class averaged nld
    neighbor_kl = compute_jsd(neighbor_dist.unsqueeze(1),connectivity_matrix.unsqueeze(0)).sum(dim=-1) # distance between node nld and each class averaged nld
    adm = (self_kl**2 + (cls_pair_jsd**2)[label[train_mask]] - neighbor_kl**2) / (2*(cls_pair_jsd**2)[label[train_mask]])

    adm[center_mask] = 0
    
    return acm, adm


def adjust_output(args, output, edge_index, label, train_mask, aggregator, class_num_list, epoch):
    """
    Adjust the margin of each labeled nodes according to local topolgy
    Input:
        args:               hyperparameters for TAM
        output:             model prediction for whole nodes (include unlabeled nodes); [# of nodes, # of classes]
        edge_index:         ; [2, # of nodes]
        label:              ; [# of nodes]
        train_mask:         ; [# of nodes]
        aggregator:         function (below)
        class_num_list:     the number of nodes for each class; [# of classes]
        epoch:              current epoch; integer
    Output:
        output:             adjusted logits
    """

    # Compute ACM and ADM
    if args.tam and epoch > args.warmup:
        acm, adm = compute_tam(output, edge_index, label, train_mask, aggregator, \
                                class_num_list=class_num_list, temp_phi = args.temp_phi, temp_gamma = 0.4)

    output = output[train_mask]
    # Adjust outputs
    if args.tam and epoch > args.warmup:
        acm = acm.log()
        adm = - adm
        output = output + args.tam_alpha*acm + args.tam_beta*adm

    return output


class MeanAggregation(MessagePassing):
    def __init__(self):
        super(MeanAggregation, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        _edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        return self.propagate(_edge_index, x=x)