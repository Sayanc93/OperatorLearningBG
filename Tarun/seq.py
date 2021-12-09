import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Reshape, Conv2D, PReLU, Flatten, Dense, Activation
from tensorflow.keras.losses import MeanSquaredError

# import matplotlib
# matplotlib.use('Agg')


class Seq_Model(tf.keras.Model):
    def __init__(self, Par):
        super(Seq_Model, self).__init__()
        # np.random.seed(23)
        tf.random.set_seed(23)

        # Defining some model parameters
        self.latent_dim = 100

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr = 10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.window_size = 200

        self.encoder = tf.keras.layers.GRU(
            20, return_sequences=True, return_state=True)

        self.decoder = tf.keras.layers.GRU(
            20, return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(
            self.window_size)

    # @tf.function()
    def call(self, X):
        X = tf.convert_to_tensor(X)

        whole_sequence_output, final_state = self.encoder(X)

        decoder_output, _ = self.decoder(
            whole_sequence_output, initial_state=final_state)

        decoder_output = tf.reshape(
            decoder_output, [tf.shape(decoder_output)[0], -1])

        y_pred = self.dense1(decoder_output)

        y_pred = tf.reshape(y_pred, [-1, self.window_size, 1])

        return(y_pred)

    # @tf.function()
    def loss(self, y_pred, y_train):

        #-------------------------------------------------------------#
        # Total Loss
        # print("Shape y_pred: ", y_pred.shape)
        # print("Shape y_train: ", y_train.shape)

        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))

        # train_loss = tf.reduce_sum(
        #     tf.cast(tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred, from_logits=True), dtype=tf.float32))
        #-------------------------------------------------------------#

        # print("train_loss: ", train_loss)

        return([train_loss])
