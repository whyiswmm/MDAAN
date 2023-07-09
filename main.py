#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.io as sio
import utils.model as Model
import torch.optim as optim
import torch.nn as nn
import math
# import connectome_conv_net_xrt.construct_brain as cb
import numpy as np
from torch.autograd import Variable
from alipy import ToolBox
from collections import Counter
from sklearn.metrics import roc_auc_score
# from utils.model import MDAAN
import warnings
warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def active_learning(select, source, target, labels):
    alibox = ToolBox(X=torch.cat((source, target), dim=0).cpu().detach().numpy(), y=labels.cpu().detach().numpy(), query_type='AllLabels', saving_path='active_learning')

    num_s, num_t = len(source), len(target)
    label_ind = [_ for _ in range(num_s)]
    unlabel_ind = [_ for _ in range(num_s, num_s+num_t)]

    uncertainStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceQUIRE', train_idx=[_ for _ in range(num_s+num_t)])

    select_ind = uncertainStrategy.select(label_ind, unlabel_ind, batch_size=1)
    select.append(select_ind[0])

def al_entro(domain_pred, num):
    inf_entro_do = []
    for i in range(4):
        entro = []
        for j in range(num):
            a, b = float(domain_pred[i][j]), float(1-domain_pred[i][j])
            entro.append(1+(a * math.log2(a) + b * math.log2(b)))
        inf_entro_do.append(entro)

    return inf_entro_do
def test_single(model, data, label):
    true_num, false_num = 0, 0
    model.eval()
    with torch.no_grad():
        _, class_out, _, domain_pred = model(data)
        # class_out = torch.sigmoid(class_out)

    num = len(label)

    for i in range(num):
        if int(class_out[i] > 0.5) == label[i]:
            true_num += 1
        else:
            false_num += 1
    return true_num

def test_mul(model, data, label):
    TP, TN, FP, FN = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        _, class_out, _, domain_pred = model(data, source=False)
        # class_out = torch.sigmoid(class_out)
        inf_entro_do = al_entro(domain_pred, len(label))

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
    acc, sen, spe, ppv, npv = (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN), TN/(TN+FP), TP/(TP+FP), TN/(TN+FN)
    bac = (sen + spe) / 2

    inf_entro_cla = []
    for i in range(num):
        a, b = float(class_out[i]), float(1 - class_out[i])
        if a == 0 or b == 0:
            inf_entro_cla.append(0)
            continue
        inf_entro_cla.append(-(a * math.log2(a) + b * math.log2(b)))

    inf_entro = [inf_entro_cla[i] + inf_entro_do[0][i] + inf_entro_do[1][i] + inf_entro_do[2][i] + inf_entro_do[3][i] for i in range(num)]

    choose_out = np.argmax(inf_entro)

    return acc, sen, spe, auc, bac, ppv, npv, choose_out

def load_schi(source, target):
    print("start test {} to {}".format(source, target))

    nets_source = []
    labels_source = []
    for dataset in source:
        nets_source.append(np.load('data/{}_cov.npy'.format(dataset)))
        labels_source.append(np.load('data/{}_labels.npy'.format(dataset)))

    nums = []
    total = 0
    for i in range(4):
        n = len(labels_source[i])
        nums.append(n)
        total += n
    para = [n/total for n in nums]

    nets_target = np.load('data/{}_cov.npy'.format(target))
    labels_target = np.load('data/{}_labels.npy'.format(target))

    return nets_source, labels_source, nets_target, labels_target, para

def load_aal(source, target):
    data = ['Leuven', 'NYU', 'UCLA', 'UM', 'USM']
    print("start test to {}".format(data[target]))
    m = sio.loadmat('data/rois_aal116.mat')
    fea = m['fea_net']
    label = m['label']
    nets_source = []
    labels_source = []
    for i in source:
        f = fea[i][0]
        tmp = np.zeros([f.shape[2],1,116,116])
        for _ in range(f.shape[2]):
            tmp[_, 0, :, :] = f[:, :, _]
        nets_source.append(tmp)
        tmp = label[i][0][0]
        for _ in range(len(tmp)):
            tmp[_] = tmp[_] - 1
        labels_source.append(tmp)

    f = fea[target][0]
    tmp = np.zeros([f.shape[2], 1, 116, 116])
    for _ in range(f.shape[2]):
        tmp[_, 0, :, :] = f[:, :, _]
    nets_target = tmp
    tmp = label[target][0][0]
    for _ in range(len(tmp)):
        tmp[_] = tmp[_] - 1
    labels_target = tmp


    return nets_source, labels_source, nets_target, labels_target

def train(source, target):
    nets_source, labels_source, nets_target, labels_target, para = load_schi(source, target)
    num = len(labels_target)

    choose_index = []     #

    'Start training'
    for shot in range(10):
        model = Model.MDAAN(DEVICE)
        model.cuda()
        '选择target数据进行训练'
        for choose in choose_index:
            tmp = [nets_target[choose, :, :, :]]
            for i in range(4):
                nets_source[i] = np.concatenate((nets_source[i], tmp), 0)
                labels_source[i] = np.append(labels_source[i], labels_target[choose])

            nets_target = np.delete(nets_target, choose, axis=0)
            labels_target = np.delete(labels_target, choose)


        source_data = []
        source_label = []
        for i in range(4):
            tmp = torch.from_numpy(nets_source[i])
            tmp = tmp.type(torch.FloatTensor).cuda()
            source_data.append(tmp.cuda())
            tmp = torch.from_numpy(labels_source[i])
            tmp = tmp.type(torch.FloatTensor)
            source_label.append(tmp.cuda())

        target_data = torch.from_numpy(nets_target)
        target_data = target_data.type(torch.FloatTensor).cuda()
        target_label = torch.from_numpy(labels_target)
        target_label = target_label.type(torch.FloatTensor)
        target_label = target_label.cuda()
        Variable(target_data)
        Variable(target_label)

        """train"""
        num_epochs = 350
        flag = 0
        lr = 0.002
        lambd = 0.5
        al_method = 'ac'    # ac or random
        max_acc, max_sen, max_spe, max_auc, max_bac, max_pev, max_npv = 0,0,0,0,0,0,0
        select = []
        loss_func = nn.BCELoss()
        for epoch in range(1,num_epochs+1):
            model.train()
            LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / (num_epochs)), 0.75)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
            loss_s_d = []
            loss_class = []
            fea_s_all = []
            for i in range(4):
                data, label = source_data[i], source_label[i]
                Variable(data), Variable(label)
                fea_s, cl_out, lo_out, _ = model(input_data=data, index=i)
                fea_s_all.append(fea_s)
                loss_class.append(loss_func(cl_out.view(len(cl_out)), label.view(len(cl_out))))
                loss_s_d.append(lo_out)


            fea_s = torch.cat((fea_s_all[0], fea_s_all[1], fea_s_all[2], fea_s_all[3]), dim=0)

            fea_t, _, loss_t_d, _ = model(input_data=target_data, source=False)

            loss_source = []
            # lambd =  lambd / math.pow((1 + 10 * (epoch - 1) / (num_epochs)), 0.75)
            for i in range(4):
                loss_source.append(lambd * para[i] * (loss_s_d[i] + loss_t_d[i]) + loss_class[i])
            loss = loss_source[0] + loss_source[1] + loss_source[2] + loss_source[3]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if flag > 3.6:
                active_learning(select=select, source=fea_s, target=fea_t,
                                labels=torch.cat((source_label[0], source_label[1], source_label[2], source_label[3], target_label), 0))
                acc, sen, spe, auc, bac, ppv, npv, test_choose = test_mul(model=model, data=target_data, label=labels_target)
                if acc > max_acc:
                    max_acc, max_sen, max_spe, max_auc, max_bac, max_pev, max_npv = acc, sen, spe, auc, bac, ppv, npv
            else:
                flag = 0
                for i in range(4):
                    data, label = source_data[i], source_label[i]
                    Variable(data), Variable(label)
                    num_src = test_single(model=model, data=data, label=label)
                    flag += num_src / len(data)
            if epoch % 500 == 0:
                print(epoch)

        if al_method == 'ac':
            counts = Counter(select).most_common(1)
            choose_index = [counts[0][0] - len(fea_s)]
        elif al_method == 'random':
            choose_index = [19]
        print(choose_index)
        print('shot:{}   & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(shot,
                                                                                                   max_acc*100,
                                                                                                   max_sen*100,
                                                                                                   max_spe*100,
                                                                                                   max_auc*100,
                                                                                                   max_bac*100,
                                                                                                   max_pev*100,
                                                                                                   max_npv*100))

def train_schizophrenia():
    datasets = ['data_huang45', 'COBRE_120_L1', 'Nottingham_68_L1', 'Taiwan_131_L1', 'Xiangya_143_L1']
    for i in range(5):
        train(source=datasets[0:i] + datasets[i+1:5], target=datasets[i])

def train_aal():
    # 0:Leuven  1:NYU   2:UCLA  3:UM    4:USM
    for i in range(1,4):
        train([_ for _ in range(5) if _ != i], i)

if __name__ == '__main__':
    train_schizophrenia()