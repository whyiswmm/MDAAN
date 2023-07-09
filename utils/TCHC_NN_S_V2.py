# -*- coding: utf-8 -*-
"""
Created on Dec.15 9:38AM 2018

Full Name of code: Temporal change of spatial correlation

@author: ML
"""
#Padding  default is valid
import numpy as np
import scipy.io as sio

from keras.models import Model
from keras import backend as K
from keras import layers

from keras.layers import Input, Flatten, LSTM
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, Average, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Reshape, Permute, Lambda

from keras import regularizers
from keras.regularizers import Regularizer
import math
import tensorflow as tf
import scipy.io as scio
class StructuredSparse(Regularizer):

    def __init__(self, C=0.001):
        self.C = K.cast_to_floatx(C)

    def __call__(self, kernel_matrix):
        return self.C * \
               K.sum(K.sqrt(K.sum(K.square(kernel_matrix), axis=-1)), axis=-1)

    def get_config(self):
        return {'C': float(self.C)}

class TCHC(object):

    #num_outputs:全连接层输出空间维度。
    #feature_depth:输出空间的维度 （即卷积中滤波器的数量）：对应的为filter 为一个序列 每一个subject一个filter
    #kernel_size：2D 卷积窗口的宽度和高度，kernel 组成filter
    #self.image_size[0]:总共的subject的个数
    #image_size:所有的fMRI的序列
    def __init__(self,
                 image_size,
                 num_chns, #16,8,1
                 num_outputs,
                 feature_depth,
                 with_bn=True,
                 with_dropout=True,
                 region_sparse=0.001,
                 drop_prob=0.5):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.img_size = image_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.region_sparse = region_sparse
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob



    def tchc_net(self, inputs, optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy']):
        conv1_convs = []
        count = 0

        for i in range(self.img_size[0]):
            conv_name = 'conv1_{0}'.format(i + 1)
            conv1 = Conv2D(filters=self.feature_depth[0],   #16
                    kernel_size=[1, 29],
                    strides=[1, 2],
                    padding='valid',
                    data_format='channels_last',
                    use_bias = False,
                    name=conv_name)(inputs[i])
            conv1 = BatchNormalization(axis=-1)(conv1)
            conv1 = Activation('relu')(conv1)
            conv1 = Dropout(self.drop_prob)(conv1)
            conv1_convs.append(conv1)
            count += 1

        conv2_convs=[]
        for i in range(self.img_size[0]):
            conv_name = 'conv2_{0}'.format(i + 1)
            conv2 = Conv2D(filters=self.feature_depth[1],   #8
                    kernel_size=[self.img_size[0]-1,1],#2
                    strides=[1, 1],
                    padding='valid',
                    use_bias=False,
                    data_format='channels_last',
                    name=conv_name)(conv1_convs[i])

            conv2 = BatchNormalization(axis=-1)(conv2)#(axis=1)
            conv2 = Activation('relu')(conv2)
            conv2 = Dropout(self.drop_prob)(conv2)
            conv2_convs.append(conv2)

        inputs_conv3= concatenate(conv2_convs,axis=1)#
        conv3 = Conv2D(filters=self.feature_depth[2],   #1
                kernel_size=[1,1],
                strides=[1, 1],
                padding='valid',
                kernel_regularizer=StructuredSparse(self.region_sparse),
                use_bias=False,
                data_format='channels_last',
                name='conv3')(inputs_conv3)

        conv3 = BatchNormalization(axis=-1)(conv3)#(axis=1)
        conv3 = Activation('relu')(conv3)

        conv3 = Dropout(self.drop_prob)(conv3)
        print()
        nd = conv3.get_shape().as_list()
        conv3_reshape = Reshape((nd[1], nd[2]))(conv3)
        print(conv3_reshape.shape)

        conv3_reshape_T = Permute((2, 1), input_shape=(nd[1], nd[2]))(conv3_reshape)


        conv3_lstm = LSTM(units=self.feature_depth[3],  #16
                          input_shape=(nd[2],nd[1]),
                          activation='tanh',
                          return_sequences=True,name='conv3_lstm')(conv3_reshape_T)

        conv3_lstm = BatchNormalization(axis=-1)(conv3_lstm)
        conv3_lstm = Dropout(self.drop_prob)(conv3_lstm)

        conv4_lstm = LSTM(units=self.feature_depth[4],activation='tanh',    #8
                          return_sequences=True,name='conv4_lstm')(conv3_lstm)
        conv4_lstm = BatchNormalization(axis=-1)(conv4_lstm)
        conv4_lstm = Dropout(self.drop_prob)(conv4_lstm)

        conv5_lstm = LSTM(units=self.feature_depth[5],activation='tanh',    #4
                          return_sequences=False,name='conv5_lstm')(conv4_lstm)
        conv5_lstm = BatchNormalization(axis=-1)(conv5_lstm)
        conv5_lstm = Dropout(self.drop_prob)(conv5_lstm)


        
        conv6_lstm = Dense(self.num_outputs, activation='sigmoid',name='conv6_lstm')(conv5_lstm)

        model = Model(inputs=inputs, outputs=[conv6_lstm])

        model.summary()

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model


