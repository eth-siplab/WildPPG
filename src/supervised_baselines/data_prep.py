import numpy as np
import os
import pickle as cp
from data_preprocess import data_preprocess_IEEE_small
from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_altx

from sklearn.metrics import f1_score
from scipy.special import softmax
import seaborn as sns
import fitlog
from copy import deepcopy

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args):
    if args.dataset == 'ieee_small':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ieee_small.prep_ieeesmall(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'altx':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180
        train_loaders, val_loader, test_loader = data_preprocess_altx.prep_altx(args)
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 190 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_dalia.prep_dalia(args)         
    return train_loaders, val_loader, test_loader
