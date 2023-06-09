#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.io as sio
import utils.model as Model
import torch.optim as optim
import torch.nn as nn
import connectome_conv_net_xrt.construct_brain as cb
import numpy as np
from torch.autograd import Variable


def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def data_make():
    """导入source数据"""
    m_1 = sio.loadmat("data/jingshenfenlie/Huaxi/AAL_TC_Huaxi_Nor.mat")
    m_2 = sio.loadmat("data/jingshenfenlie/Huaxi/AAL_TC_Huaxi_Pat.mat")
    name = [_ for _ in m_1]
    print(name)
    result_1 = m_1['AAL_TC_Huaxi_Nor']
    result_2 = m_2['AAL_TC_Huaxi_Pat']

    labels_source = labels[0, :]
    for i in range(len(labels_source)):
        if labels_source[i] == -1:
            labels_source[i] = 0
    num = result.shape[0]

    'source'
    '构建脑网络'
    subjects_net_source_1 = cb.static_brain_construct(result, num, 'correlation')
    subjects_net_source_2 = cb.static_brain_construct(result, num, 'covariance')
    subjects_net_source_2 = normalization(subjects_net_source_2)
    """构建DTW网络"""
    dtw_nets_source = cb.create_dtw_static_net(result, 1, num)
    np.save('./dtw_Nottingham_68.npy', dtw_nets_source)
    dtw_nets_source = np.load('dtw_COBRE.npy')
    """组织成两个通道的网络"""
    nets_source = np.zeros((num, 2, 90, 90))
    for i in range(num):
        nets_source[i, 0, :, :] = subjects_net_source_1[i, :, :]
        nets_source[i, 1, :, :] = subjects_net_source_2[i, :, :]
        nets_source[i, 2, :, :] = dtw_nets_source[i, :, :]
    np.save('./data/{}.npy'.format(source), nets_source)

data_make()