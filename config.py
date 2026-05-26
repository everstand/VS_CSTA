import argparse
import numpy as np
import random
import torch

VALID_DATASETS = ('SumMe','TVSum')

def str2datasets(v):
    if isinstance(v,(list,tuple)):
        datasets = list(v)
    else:
        datasets = [item.strip() for item in str(v).split(',') if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError('datasets must contain SumMe, TVSum, or both')
    invalid = [item for item in datasets if item not in VALID_DATASETS]
    if invalid:
        raise argparse.ArgumentTypeError(f'Unknown dataset(s): {invalid}. Valid values are: {VALID_DATASETS}')
    return datasets

# Process bool argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise

# Process none argument
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return v

# Define configuration class
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.datasets = str2datasets(getattr(self,'datasets',['SumMe','TVSum']))
        self.SumMe_len = 25
        self.TVSum_len = 50

        # Set device
        if self.device!='cpu':
            torch.cuda.set_device(self.device)

        # Set seed
        self.set_seed()

    # Set the seed
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device!='cpu':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Define all configurations
def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str2datasets, default=['SumMe','TVSum'])
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', default='1')
    parser.add_argument('--learning_rate', default='1e-3')
    parser.add_argument('--weight_decay', default='1e-7')

    # Optional experiment modules. Defaults preserve the clean CSTA baseline.
    parser.add_argument('--use_multi_annot_loss', type=str2bool, default=False)
    parser.add_argument('--ma_precision_ema', type=float, default=0.9)
    parser.add_argument('--ma_precision_min', type=float, default=0.1)
    parser.add_argument('--ma_precision_max', type=float, default=10.0)
    parser.add_argument('--ma_precision_warmup', type=int, default=0)
    parser.add_argument('--ma_loss_weight', type=float, default=1.0)
    parser.add_argument('--ma_loss_weight_summe', type=str2none, default=None)
    parser.add_argument('--ma_loss_weight_tvsum', type=str2none, default=None)
    parser.add_argument('--use_rank_loss', type=str2bool, default=False)
    parser.add_argument('--rank_loss_weight', type=float, default=0.0)
    parser.add_argument('--rank_loss_min_diff', type=float, default=0.0)
    parser.add_argument('--use_shot_loss', type=str2bool, default=False)
    parser.add_argument('--shot_loss_weight', type=float, default=0.0)
    parser.add_argument('--shot_loss_min_diff', type=float, default=0.0)
    
    parser.add_argument('--model_name', type=str, default='GoogleNet_Attention')
    parser.add_argument('--Scale', type=str2none, default=None)
    parser.add_argument('--Softmax_axis', type=str2none, default='TD')
    parser.add_argument('--Balance', type=str2none, default=None)

    parser.add_argument('--Positional_encoding', type=str2none, default='FPE')
    parser.add_argument('--Positional_encoding_shape', type=str2none, default='TD')
    parser.add_argument('--Positional_encoding_way', type=str2none, default='PGL_SUM')
    parser.add_argument('--Dropout_on', type=str2bool, default=True)
    parser.add_argument('--Dropout_ratio', default='0.6')

    parser.add_argument('--Classifier_on', type=str2bool, default=True)
    parser.add_argument('--CLS_on', type=str2bool, default=True)
    parser.add_argument('--CLS_mix', type=str2none, default='Final')

    parser.add_argument('--key_value_emb', type=str2none, default='kv')
    parser.add_argument('--Skip_connection', type=str2none, default='KC')
    parser.add_argument('--Layernorm', type=str2bool, default=True)

    # Generate summary videos
    parser.add_argument('--input_is_file', type=str2bool, default='true')
    parser.add_argument('--file_path', type=str, default='./SumMe/Jumps.mp4')
    parser.add_argument('--dir_path', type=str, default='./SumMe')
    parser.add_argument('--ext', type=str, default='mp4')
    parser.add_argument('--sample_rate', type=int, default=15)
    parser.add_argument('--save_path', type=str, default='./summary_videos')
    parser.add_argument('--weight_path', type=str, default='./weights/SumMe/split4.pt')

    kwargs = vars(parser.parse_args())
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
