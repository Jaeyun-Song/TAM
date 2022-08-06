import torch
import numpy as np
from torch_scatter import scatter_add


def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == "chameleon" or name == "squirrel":
        from dataset.WikipediaNetwork import WikipediaNetwork
        dataset = WikipediaNetwork(path, name, transform = T.NormalizeFeatures())
    elif name == "Wisconsin":
        # from torch_geometric.datasets import  WebKB
        from dataset.WebKB import WebKB
        dataset = WebKB(path, name, transform = T.NormalizeFeatures())
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset


def split_semi_dataset(total_node, n_data, n_cls, class_num_list, idx_info, device):
    new_idx_info = []
    _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool, device=device)
    for i in range(n_cls):
        if n_data[i] > class_num_list[i]:
            cls_idx = torch.randperm(len(idx_info[i]))
            cls_idx = idx_info[i][cls_idx]
            cls_idx = cls_idx[:class_num_list[i]]
            new_idx_info.append(cls_idx)
        else:
            new_idx_info.append(idx_info[i])
        _train_mask[new_idx_info[i]] = True

    assert _train_mask.sum().long() == sum(class_num_list)
    assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

    return _train_mask, new_idx_info


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info
