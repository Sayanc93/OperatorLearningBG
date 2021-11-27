# -*- coding: utf-8 -*-
# """
# Created on Sat Jun  5 21:47:39 2021

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Reshape, Conv2D, PReLU, Flatten, Dense, Activation
from tensorflow.keras.losses import MeanSquaredError


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import os
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#os.environ["CUDA_VISIBLE_DEVICES"]='0'


class DeepONet_Model(tf.keras.Model):

    def __init__(self, Par):
        super(DeepONet_Model, self).__init__()
        np.random.seed(23)
        tf.random.set_seed(23)

        #Defining some model parameters
        self.latent_dim = 100

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.branch_net = Sequential()
        self.branch_net.add(Dense(100, activation='relu'))
        self.branch_net.add(Dense(100, activation='relu'))
        self.branch_net.add(Dense(self.latent_dim))


        self.trunk_net  = Sequential()
        self.trunk_net.add(Dense(100, activation='relu'))
        self.trunk_net.add(Dense(100, activation='relu'))
        self.trunk_net.add(Dense(self.latent_dim))

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc):
    #X_func -> [BS, m]
    #X_loc  -> [npoints_output, 1]

        y_func = self.branch_net(X_func)

        y_loc = self.trunk_net(X_loc)

        Y = tf.einsum('ij,kj->ik', y_func, y_loc)

        return(Y)

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):

        #-------------------------------------------------------------#
        #Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        #-------------------------------------------------------------#

        return([train_loss])
