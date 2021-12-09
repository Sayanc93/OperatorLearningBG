# -*- coding: utf-8 -*-
# """
# @author: Sayan Chakraborty
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import os
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

tf.config.optimizer.set_jit(True)

class LSTM_Model(Model):

    def __init__(self, Par):
        super(LSTM_Model, self).__init__()
        np.random.seed(23)
        tf.random.set_seed(23)

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.model = Sequential()
        self.model.add(LSTM(20))
        self.model.add(Dense(20, activation='relu'))

    @tf.function()
    def call(self, X):
        X = tf.convert_to_tensor(X)
        pred = self.model(X)
        return tf.reshape(pred, X.shape)

    @tf.function()
    def loss(self, y_pred, y_train):

        #-------------------------------------------------------------#
        #Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        #-------------------------------------------------------------#

        return([train_loss])
