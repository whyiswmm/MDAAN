from keras.models import Model
from keras.layers.convolutional import Conv2D


class my_model(object):

    def __init__(self, class_num=2):
        self.num_classes=class_num

    def model_test(self):

        conv1 = Conv2D(filters=64, kernel_size=5)