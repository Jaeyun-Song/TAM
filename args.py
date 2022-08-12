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
    parser.add_argument('--loss_type', type=str, default='bs',
                        help='Loss type')
    # Method
    parser.add_argument('--tam', action='store_true',
                        help='use tam')
    parser.add_argument('--reweight', action='store_true',
                        help='use reweight')
    parser.add_argument('--pc_softmax', action='store_true',
                        help='use pc softmax')
    parser.add_argument('--ens', action='store_true',
                        help='use GraphENS')
    parser.add_argument('--renode', action='store_true',
                        help='use ReNode')
    # Hyperparameter for GraphENS
    parser.add_argument('--keep_prob', type=float, default=0.01,
                        help='Keeping Probability')
    parser.add_argument('--pred_temp', type=float, default=2,
                        help='Prediction temperature')             
    # ReNode
    parser.add_argument('--loss_name', default="re-weight", type=str, help="the training loss") #ce focal re-weight cb-softmax
    parser.add_argument('--factor_focal', default=2.0,    type=float, help="alpha in Focal Loss")
    parser.add_argument('--factor_cb',    default=0.9999, type=float, help="beta  in CB Loss")
    parser.add_argument('--rn_base',    default=0.5, type=float, help="Lower bound of RN")
    parser.add_argument('--rn_max',    default=1.5, type=float, help="Upper bound of RN")

    # Hyperparameter for TAM
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