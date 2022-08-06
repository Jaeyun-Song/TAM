import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='CiteSeer',
                        help='Dataset Name')
    parser.add_argument('--imb_ratio', type=float, default=10,
                        help='Imbalance Ratio')
    # Architecture
    parser.add_argument('--net', type=str, default='GCN',
                        help='Architecture name')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension')
    # Imbalance Loss
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss type')
    # Method
    parser.add_argument('--tam', action='store_true',
                        help='use tam')
    parser.add_argument('--reweight', action='store_true',
                        help='use reweight')
    parser.add_argument('--pc_softmax', action='store_true',
                        help='use pc softmax')
    # Hyperparameter for our approach
    parser.add_argument('--tam_alpha', type=float, default=2.5,
                        help='coefficient of ACM')
    parser.add_argument('--tam_beta', type=float, default=0.5,
                        help='coefficient of ADM')
    parser.add_argument('--temp_phi', type=float, default=1.2,
                        help='classwise temperature')
    parser.add_argument('--warmup', type=int, default=5,
                        help='warmup')
    args = parser.parse_args()

    return args