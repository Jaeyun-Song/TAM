"""
Our code is based on GraphENS:
https://github.com/JoonHyung-Park/GraphENS
"""

import os.path as osp
import random
import torch
import torch.nn.functional as F
from nets import *
from data_utils import *
from args import parse_args
from models import *
from losses import *
from sklearn.metrics import balanced_accuracy_score, f1_score
import statistics
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Arg Parser ##
args = parse_args()

## Handling exception from arguments ##
assert not (args.warmup < 1 and args.tam)
# assert args.imb_ratio > 1

## Load Dataset ##
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = get_dataset(dataset, path, split_type='public')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)


def train():
    global class_num_list, aggregator
    global data_train_mask, data_val_mask, data_test_mask

    model.train()
    optimizer.zero_grad()        

    output = model(data.x, data.edge_index)
        
    criterion(output[data_train_mask], data.y[data_train_mask], weight=class_weight).backward()

    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index)
        val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])

    optimizer.step()
    scheduler.step(val_loss)


@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    accs, baccs, f1s = [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)

    return accs, baccs, f1s


## Log for Experiment Setting ##
setting_log = "Dataset: {}, ratio: {}, net: {}, n_layer: {}, feat_dim: {}, tam: {}".format(
    args.dataset, str(args.imb_ratio), args.net, str(args.n_layer), str(args.feat_dim), str(args.tam))

repeatition = 10
seed = 100
avg_val_acc_f1, avg_test_acc, avg_test_bacc, avg_test_f1 = [], [], [], []
for r in range(repeatition):

    ## Fix seed ##
    torch.cuda.empty_cache()
    seed += 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    if args.dataset in ['squirrel', 'chameleon', 'Wisconsin']:
        data_train_mask, data_val_mask, data_test_mask = data.train_mask[:,r%10].clone(), data.val_mask[:,r%10].clone(), data.test_mask[:,r%10].clone()
    else:
        data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()

    ## Data statistic ##
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

    # for artificial imbalanced setting: only the last imb_class_num classes are imbalanced
    imb_class_num = n_cls // 2
    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])
    for i in range(n_cls):
        if args.imb_ratio > 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
            new_class_num_list.append(min(int(max_num*(1./args.imb_ratio)), class_num_list[i]))
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list

    if args.imb_ratio > 1:
        data_train_mask, idx_info = split_semi_dataset(len(data.x), n_data, n_cls, class_num_list, idx_info, data.x.device)

    ## Re-weight method ##
    class_weight = get_weight(args.reweight, class_num_list).to(device)

    ## Model Selection ##
    if args.net == 'GCN':
        model = create_gcn(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == 'GAT':
        model = create_gat(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == "SAGE":
        model = create_sage(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    else:
        raise NotImplementedError("Not Implemented Architecture!")

    ## Criterion Selection ##
    if args.loss_type == 'ce': # CE
        criterion = CrossEntropy()
    else:
        raise NotImplementedError("Not Implemented Loss!")

    model = model.to(device)
    criterion = criterion.to(device)

    # Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0),], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.5,
                                                           patience = 100,
                                                           verbose=False)

    # Train models
    best_val_acc_f1 = 0
    aggregator = MeanAggregation()
    for epoch in range(1, 2001):

        train()
        accs, baccs, f1s = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, val_f1, tmp_test_f1 = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        if val_acc_f1 > best_val_acc_f1:
            best_val_acc_f1 = val_acc_f1
            test_acc = accs[2]
            test_bacc = baccs[2]
            test_f1 = f1s[2]

    avg_val_acc_f1.append(best_val_acc_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)

## Calculate statistics ##
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)
avg_val_acc_f1 = statistics.mean(avg_val_acc_f1)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc F1: {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc_f1)
log = "{}\n{}".format(setting_log, avg_log)
print(log)