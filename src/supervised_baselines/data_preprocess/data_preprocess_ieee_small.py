'''
Data Pre-processing on ieeesmall dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split, train_val_split
from data_preprocess.base_loader import base_loader
from torchvision import transforms


def load_domain_data(domain_idx):
    str_folder = './data/'
    data_all = scipy.io.loadmat(str_folder + 'IEEE.mat')
    data = data_all['whole_dataset']
    domain_idx = int(domain_idx)
    X = data[domain_idx,0]
    y = np.squeeze(data[domain_idx,1]) - 1
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y, d

class data_loader_ieeesmall(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_ieeesmall, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return np.squeeze(np.transpose(sample, (1, 0, 2)),0), target, domain

def prep_domains_ieeesmall_subject_sp(args):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 200, 1)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    x_win_train, x_win_val, \
    y_win_train, y_win_val, \
    d_win_train, d_win_val = train_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_train, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ieeesmall(x_win_train, y_win_train, d_win_train)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    source_loaders = [source_loader]

    ### 
    data_set_val = data_loader_ieeesmall(x_win_val, y_win_val, d_win_val)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = val_loader   

    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 200, 1)), (0, 2, 1, 3))


    data_set = data_loader_ieeesmall(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

def prep_ieeesmall(args):
    if args.cases == 'subject_val':
        return prep_domains_ieeesmall_subject_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

