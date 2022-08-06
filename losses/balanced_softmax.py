"""
Code Reference: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py
"""

"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.
Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, cls_num):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(cls_num)

    def forward(self, input, label, reduction='mean', weight=None):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss