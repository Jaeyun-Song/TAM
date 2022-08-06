import torch
import torch.nn.functional as F

def pc_softmax(logits, cls_num):
    sample_per_class = torch.tensor(cls_num)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits - spc.log()
    return logits