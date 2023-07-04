import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import cross_transformer 
from models import conv1d
from models import conv2d
from models import conv3d
import utils
from utils import recorder
from evaluation import HSIEvaluation
from augment import do_augment
import itertools


def func_multi_linear(cur_epoch):
    '''
        分段线性
    '''
    return 0.1

    if cur_epoch < 100:
        weight = 0
    else:
        weight = min(float(cur_epoch - 60) / 10, 0.1)
    return weight


def func_soft_linear(cur_epoch):
    '''
        纯线性
    '''
    weight = min(float(cur_epoch) / 30, 1.0)
    return weight
    

def get_fun(sign='multi_linear'):
    if sign == 'multi_linear':
        return func_multi_linear
    elif sign == 'soft_linear':
        return func_soft_linear
    else:
        assert "not Implemented Error"



def func_label_multi_linear(cur_epoch):
    '''
        分段线性
    '''
    if cur_epoch < 100:
        weight = 0
    else:
        weight = 1
    return weight


def func_label_soft_linear(cur_epoch):
    '''
        纯线性
    '''
    weight = min(float(cur_epoch) / 30, 1.0)
    return weight
    

def get_fun_label(sign='label_multi_linear'):
    if sign == 'label_multi_linear':
        return func_label_multi_linear
    elif sign == 'label_soft_linear':
        return func_label_soft_linear
    else:
        assert "not Implemented Error"