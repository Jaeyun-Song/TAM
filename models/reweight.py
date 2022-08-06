import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_weight(is_reweight, class_num_list):
    if is_reweight:
        min_number = np.min(class_num_list)
        class_weight_list = [float(min_number)/float(num) for num in class_num_list]
    else:
        class_weight_list = [1. for _ in class_num_list]
    class_weight = torch.tensor(class_weight_list).type(torch.float32)

    return class_weight