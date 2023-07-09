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

# class DTN(nn.Module):
#     def __init__(self):
#         super(DTN, self).__init__()
#         self.conv_params = nn.Sequential (
#                 nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(64),
#                 nn.Dropout2d(0.1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(128),
#                 nn.Dropout2d(0.3),
#                 nn.ReLU(),
#                 nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(256),
#                 nn.Dropout2d(0.5),
#                 nn.ReLU()
#                 )
#
#     def forward(self, x):
#         return self.conv_params(x)
#
# class DTNclassifier(nn.Module):
#     def __init__(self):
#         super(DTNclassifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(256*12*12, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(512,2)
#         )
#
#     def forward(self, x):
#         return self.classifier(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128*1*1, 96))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout(p=0.6))
        self.class_classifier.add_module('c_fc2', nn.Linear(96, 1))

    def forward(self, x):
        x = self.class_classifier(x)
        x = torch.sigmoid(x)
        return x


class MDAAN(nn.Module):

    def __init__(self, device):
        super(MDAAN, self).__init__()
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
            loss_adv = []
            domain_pred = []
            for i in range(4):
                domain_pred.append(self.domain_classifier[i](x))
                loss_adv.append(loss_fn(domain_pred[i], domain_label.float()))
        return loss_adv, domain_pred



#
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }
#
#
# #分类网络
# class VGG(nn.Module):
#     def __init__(self, features, num_classes=29, init_weights=False):
#         super(VGG, self).__init__()
#         self.features = features
#
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.features(x)
#         # N x 512 x 7 x 7
#         # x = torch.flatten(x, start_dim=1)#展平处理
#         # # N x 512*7*7
#         # x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self): #初始化权重函数
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)#初始化偏置为0
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 # nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# def make_features(cfg: list):#提取特征函数
#     layers = [] #存放创建每一层结构
#     in_channels = 3 #RGB
#     for v in cfg:
#         if v == "M":
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(True)]
#             in_channels = v
#     return nn.Sequential(*layers)   #通过非关键字参数传入
#
#
# class VGGNET(nn.Module):
#
#     def __init__(self, model_name='vgg16', pretrained=False, classes=2):
#         super(VGGNET, self).__init__()
#         self.featrues = vgg(model_name=model_name, pretrained=pretrained)
#         self.classifier = nn.Sequential( #定义分类网络结构
#             nn.Dropout(p=0.5), #减少过拟合
#             nn.Linear(512*7*7, 2048),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(True),
#             nn.Linear(2048, classes)
#         )
#
#     def forward(self, x):
#         x = self.featrues(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)
#
#         return x
#
#
#
# cfgs = {
#     'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #数字为卷积层个数，M为池化层结构
#     'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
# #实例化VGG网络
# def vgg(model_name="vgg16", pretrained=False):
#     try:
#         cfg = cfgs[model_name]
#     except:
#         print("Warning: model number {} not in cfgs dict!".format(model_name))
#         exit(-1)
#     model = VGG(make_features(cfg))
#
#     if pretrained:
#         pretrained_state_dict = model_zoo.load_url(model_urls['vgg16'])
#         now_state_dict = model.state_dict()
#         now_state_dict.update(pretrained_state_dict)
#         model.load_state_dict(now_state_dict, strict=False)
#     return model