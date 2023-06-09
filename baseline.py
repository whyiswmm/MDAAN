#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.io as sio
import utils.baseline_model as Model
import torch.optim as optim
import torch.nn as nn
import math
import connectome_conv_net_xrt.construct_brain as cb
import numpy as np
from torch.autograd import Variable
from alipy import ToolBox
from collections import Counter
from sklearn.metrics import roc_auc_score
from utils.model import DANN

DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

seed = 0
print('seed is {}'.format(seed))
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test_single(model, data, label):
    true_num, false_num = 0, 0
    model.eval()
    with torch.no_grad():
        class_out, domain_pred = model(data)

    num = len(label)

    for i in range(num):
        if int(class_out[i] > 0.5) == label[i]:
            true_num += 1
        else:
            false_num += 1

    return true_num

def test_mul(model, data, label):
    TP, TN, FP, FN = 0, 0, 0, 0
    acc, sen, spe, auc, bac, ppv, npv = 0,0,0,0,0,0,0
    model.eval()
    with torch.no_grad():
        class_out, domain_pred = model(data, source=False)

    num = len(label)

    for i in range(num):
        if class_out[i] > 0.5:
            if label[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[i] == 0:
                TN += 1
            else:
                FN += 1
    auc = roc_auc_score(label, class_out.cpu().numpy())
    try:
        acc, sen, spe, ppv, npv = (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN), TN/(TN+FP), TP/(TP+FP), TN/(TN+FN)
        bac = (sen + spe) / 2
    except ZeroDivisionError:
        pass

    inf_entro_cla = []
    for i in range(num):
        a, b = float(class_out[i]), float(1 - class_out[i])
        if a == 0 or b == 0:
            inf_entro_cla.append(0)
            continue
        inf_entro_cla.append(-(a * math.log2(a) + b * math.log2(b)))


    return acc, sen, spe, auc, bac, ppv, npv, np.argmax(inf_entro_cla)


def train(source, target):
    print("start test {} to {}".format(source, target))

    # nets_source = np.load('data/{}_cov.npy'.format(source))
    # labels_source = np.load('data/{}_labels.npy'.format(source))

    nets_source = np.load('data/{}_cov.npy'.format(source[0]))
    labels_source = np.load('data/{}_labels.npy'.format(source[0]))
    for dataset in source[1:]:
        nt = np.load('data/{}_cov.npy'.format(dataset))
        lt = np.load('data/{}_labels.npy'.format(dataset))
        nets_source = np.concatenate((nets_source, nt), 0)
        labels_source = np.concatenate((labels_source, lt), 0)

    nets_target = np.load('data/{}_cov.npy'.format(target))
    labels_target = np.load('data/{}_labels.npy'.format(target))

    'Start training'
    model = Model.DANN(DEVICE)
    # model.load_state_dict(torch.load('model.pth'))
    model.cuda()
    '选择target数据进行训练'

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    choose_index = []
    for shot in range(5):

        for choose in choose_index:
            tmp = [nets_target[choose, :, :, :]]
            nets_source = np.concatenate((nets_source, tmp), 0)
            labels_source = np.append(labels_source, labels_target[choose])

            nets_target = np.delete(nets_target, choose, axis=0)
            labels_target = np.delete(labels_target, choose)

        source_data = torch.from_numpy(nets_source)
        source_data = source_data.type(torch.FloatTensor)
        source_data = source_data.cuda()
        source_label = torch.from_numpy(labels_source)
        source_label = source_label.type(torch.FloatTensor)
        source_label = source_label.cuda()
        target_data = torch.from_numpy(nets_target)
        target_data = target_data.type(torch.FloatTensor).cuda()
        Variable(source_data)
        Variable(source_label)
        Variable(target_data)

        """train"""
        num_epochs = 500
        num_src = 0
        select = []
        max_acc, max_sen, max_spe, max_auc, max_bac, max_pev, max_npv = 0,0,0,0,0,0,0
        loss_func = nn.BCELoss()
        for epoch in range(num_epochs):
            model.train()

            class_out, loss_s_d = model(input_data=source_data)
            loss_class = loss_func(class_out.view(len(class_out)), source_label)
            _, loss_t_d = model(input_data=target_data, source=False)

            loss_domain = loss_s_d + loss_t_d
            loss = 0.2 * loss_domain + loss_class

            # print('epoch:{}  loss:{}  class loss:{}  domain loss:{}'.format(epoch, loss, loss_class, loss_domain))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if num_src/len(source_label) > 0.9:
                acc, sen, spe, auc, bac, ppv, npv, ch = test_mul(model=model, data=target_data, label=labels_target)
                select.append(ch)
                if acc > max_acc:
                    max_acc, max_sen, max_spe, max_auc, max_bac, max_pev, max_npv = acc, sen, spe, auc, bac, ppv, npv
            else:
                num_src = test_single(model=model, data=source_data, label=source_label)
        counts = Counter(select).most_common(2)
        choose_index = [counts[0][0]]
        print('shot:{} choose:{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(   shot,
                                                                                                                choose_index,
                                                                                                                max_acc*100,
                                                                                                                max_sen*100,
                                                                                                                max_spe*100,
                                                                                                                max_auc*100,
                                                                                                                max_bac*100,
                                                                                                                max_pev*100,
                                                                                                                max_npv*100))

if __name__ == '__main__':
    datasets = ['data_huang45', 'COBRE_120_L1', 'Nottingham_68_L1', 'Taiwan_131_L1', 'Xiangya_143_L1']
    for i in range(5):
        train(source=datasets[0:i] + datasets[i+1:5], target=datasets[i])
    # for target in datasets:
    #     for source in datasets:
    #         if source!=target:
    #             train(source, target)