import torch
import torch.nn as nn
from torch.autograd import Variable
import utils.adv_layer as adv_layer
import torch.utils.model_zoo as model_zoo

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1,90), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Conv2d(64, 128, kernel_size=(90, 1), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.6),
        )



    def forward(self, x):
        return self.feature(x)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_drop1', nn.Dropout(p=0.6))
        self.class_classifier.add_module('c_fc1', nn.Linear(128*1*1, 96))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout(p=0.6))
        self.class_classifier.add_module('c_fc2', nn.Linear(96, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 1))

        # self.classifier = nn.Sequential( #定义分类网络结构
        #     nn.Dropout(p=0.5), #减少过拟合
        #     nn.Linear(512*2*2, 2048),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, 2)
        # )

    def forward(self, x):
        x = self.class_classifier(x)
        x = torch.sigmoid(x)
        return x


class DANN(nn.Module):

    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.domain_classifier = []
        for i in range(4):
            self.domain_classifier.append(adv_layer.Discriminator(
                input_dim=128 * 1 * 1, hidden_dim=96).cuda())

    def forward(self, input_data, alpha=1, index=0, source=True):
        # input_data = input_data.expand(len(input_data), 3, 28, 28)
        feature = self.feature(input_data)
        feature = torch.flatten(feature, start_dim=1)
        class_output = self.classifier(feature)
        domain_loss, domain_pred = self.get_adversarial_result(
            feature, source, index, alpha)
        return feature, class_output, domain_loss, domain_pred

    def get_adversarial_result(self, x, source=True, index=0, alpha=1):
        loss_fn = nn.BCELoss()  #只用于二分类的loss
        if source:
            domain_label = torch.ones(len(x)).long().cuda()
            x = adv_layer.ReverseLayerF.apply(x, alpha)
            domain_pred = self.domain_classifier[index](x)
            loss_adv = loss_fn(domain_pred, domain_label.float())
        else:
            domain_label = torch.zeros(len(x)).long().cuda()
            x = adv_layer.ReverseLayerF.apply(x, alpha)
            domain_pred = self.domain_classifier[index](x)
            loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv, domain_pred