import torch
import torch.nn as nn



class STNet(nn.Module):

    def __init__(self, image_size, feature_depth):
        super(STNet, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('c1', nn.Conv2d(1, feature_depth[0], kernel_size=(1, 30), stride=(1,2)))
        self.conv1.add_module('b1', nn.BatchNorm2d(feature_depth[0]))
        self.conv1.add_module('r1', nn.ReLU(True))
        self.conv1.add_module('d1', nn.Dropout2d(p=0.5))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('c2', nn.Conv2d(16, feature_depth[1], kernel_size=(89, 1), stride=1))
        self.conv2.add_module('b2', nn.BatchNorm2d(feature_depth[1]))
        self.conv2.add_module('r2', nn.ReLU(True))
        self.conv2.add_module('d2', nn.Dropout2d(p=0.5))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('c3', nn.Conv2d(8, feature_depth[2], kernel_size=(1,1), stride=1))
        self.conv3.add_module('b3', nn.BatchNorm2d(feature_depth[2]))
        self.conv3.add_module('r3', nn.ReLU())
        self.conv3.add_module('d3', nn.Dropout2d(p=0.5))

        self.image_size = image_size
        self.feature_depth = feature_depth


    def forward(self, input):
        conv1_convs = []
        count = 0

        for i in range(self.image_size[0]):
            conv1 = self.conv1(input[i])
            conv1_convs.append(conv1)
            count += 1

        conv2_convs = []
        for i in range(self.image_size[0]):
            conv2 = self.conv2(conv1_convs[i])
            conv2_convs.append(conv2)

        inputs_conv3 = torch.cat(conv2_convs, 3)
        conv3 = self.conv3(inputs_conv3)
        # #此处应有一个正则化，暂时先不写

        # #此处操作为调整卷积层维度，具体等调整的时候再写
        nd = conv3.size()
        print(nd)
        # conv3_reshape = torch.reshape(conv3, [nd[1], nd[2]])    #具体是啥还得调

        return conv3