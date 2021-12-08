import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Reshape, Conv2D, PReLU, Flatten, Dense, Activation
from tensorflow.keras.losses import MeanSquaredError

import matplotlib
matplotlib.use('Agg')


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

        # self.embedding_size = 128
        self.window_size = 20

        # self.encoder_embed = tf.keras.layers.Embedding(
        #     150, self.embedding_size, input_length=self.window_size)
        # self.decoder_embed = tf.keras.layers.Embedding(
        #     150, self.embedding_size, input_length=self.window_size)

        self.encoder = tf.keras.layers.GRU(
            self.window_size, return_sequences=True, return_state=True)

        self.decoder = tf.keras.layers.GRU(
            self.window_size, return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(
            self.window_size)

        # self.dense2 = tf.keras.layers.Dense(
        #     1)

    # @tf.function()
    def call(self, X):
        X = tf.convert_to_tensor(X)

        # print("X shape: ", X.shape)

        # encoder_embeddings = self.encoder_embed(X_func)

        # print("encoder_embeddings shape: ", encoder_embeddings.shape)
        # print("encoder_embeddings: ", encoder_embeddings[0, 0, 0])

        # X = tf.reshape(X, [-1, self.window_size, 1])

        # print("X shape: ", X.shape)

        whole_sequence_output, final_state = self.encoder(X)

        # print("encoder_output whole_sequence_output shape: ", whole_sequence_output.shape)
        # print("encoder_output final_state shape: ", final_state.shape)

        # decoder_embeddings = self.decoder_embed(y)

        # print("decoder_embeddings shape: ", decoder_embeddings.shape)

        decoder_output, _ = self.decoder(
            whole_sequence_output, initial_state=final_state)

        decoder_output = tf.reshape(
            decoder_output, [tf.shape(decoder_output)[0], -1])

        # print("decoder_output shape: ", decoder_output.shape)

        # logits1 = self.dense1(decoder_output)

        y_pred = self.dense1(decoder_output)

        y_pred = tf.reshape(y_pred, [-1, self.window_size, 1])

        # print("y_pred shape: ", y_pred.shape)

        # option 1 : Return decoder_output as y_pred (batch_size, window_size, 1)

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
