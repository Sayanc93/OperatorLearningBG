import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from lstm import LSTM_Model
from gru import GRU_Model
from os import sys


def preprocess(dataset, m, npoints_output, is_test=False):
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m))  # [1500, m]
    # X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y = R(np.linspace(t_min, t_max, m))  # [1500, npoints_output]

    X_func = tf.convert_to_tensor(X_func)
    y = tf.convert_to_tensor(y)

    if is_test:
        X_func = X_func[:50]
        y = y[:50]

    return X_func, y


def normalize(X_func, y, Par):
    X_func = (X_func - Par['p_mean'])/Par['p_std']
    y = (y - Par['r_mean'])/Par['r_std']

    return tf.cast(X_func, tf.float32), tf.cast(y, tf.float32)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"LSTM","GRU"} or sys.argv[2] == "":
        print("USAGE: python train.py <Model Type> <M Value>")
        print("<Model Type>: [LSTM/GRU]")
        print("<M Value>: 20/80/200")
        exit()

    Par = {}
    Par['model'] = 'lstm' if sys.argv[1] == "LSTM" else 'gru'

    train_dataset = np.load('../data/0.1/res_1000.npz')
    test_dataset = np.load('../data/0.08/res_1000.npz')

    Par['m'] = int(sys.argv[2])
    m = Par['m']
    npoints_output = 200
    Par['address'] = 'lstm_'+str(m) if sys.argv[1] == "LSTM" else 'gru_'+str(m)

    X_train, y_train = preprocess(
        train_dataset, m, npoints_output)
    X_test, y_test = preprocess(
        test_dataset, m, npoints_output, is_test=True)

    X_train = tf.reshape(X_train, [-1, m, 1])
    X_test = tf.reshape(X_test, [-1, m, 1])
    y_train = tf.reshape(y_train, [-1, m, 1])
    y_test = tf.reshape(y_test, [-1, m, 1])

    Par['p_mean'] = tf.math.reduce_mean(X_train)
    Par['p_std'] = tf.math.reduce_std(X_train)

    Par['r_mean'] = tf.math.reduce_mean(y_train)
    Par['r_std'] = tf.math.reduce_std(y_train)

    # plt.plot(range(len(X_train[0])), X_train[0])
    # plt.show()

    X_train, y_train = normalize(X_train, y_train, Par)
    X_test, y_test = normalize(X_test, y_test, Par)

    # print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    # print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)

    model = LSTM_Model(Par) if sys.argv[1] == "LSTM" else GRU_Model(Par)
    model_address = Par['model'] + '_' + str(Par['m']) + '/model_20000'
    model.load_weights(model_address)

    for i in [0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print('\nGRF(l={:.1})'.format(i))
        print('-'*20)

        address = '../data/'+str(i)+'/res_1000.npz'
        # address = 'data/'+str(i)+'/res_1000.npz'
        test_dataset = np.load(address)

        npoints_output = 500

        X_test, y_test = preprocess(
            test_dataset, m, npoints_output, is_test=True)

        X_truth, y_truth = preprocess(
            test_dataset, 1000, npoints_output, is_test=True)
        X_test, y_test = normalize(X_test[:1], y_test[:1], Par)
        X_truth, y_truth = normalize(X_truth[:1], y_truth[:1], Par)

        print('truth:', y_truth.shape)

        X_test = tf.reshape(X_test, [-1, m, 1])
        y_test = tf.reshape(y_test, [-1, m, 1])

        y_pred = model(X_test)
        mse = tf.math.reduce_mean((y_pred - y_test)**2)
        print('MSE: {:.4e}'.format(mse))

        y_pred = tf.cast(y_pred, tf.float64)
        y_test = tf.cast(y_test, tf.float64)
        y_truth = tf.cast(y_truth, tf.float64)

        y_pred = y_pred*Par['r_std'] + Par['r_mean']
        y_test = y_test*Par['r_std'] + Par['r_mean']
        y_truth = y_truth*Par['r_std'] + Par['r_mean']

        plt.figure(figsize=(10, 10))
        plt.plot(np.ravel(np.linspace(0, 5*10**-4, m)),
                 np.ravel(y_pred), label=str(Par['model']))
        plt.plot(np.ravel(np.linspace(0, 5*10**-4, 1000)),
                 np.ravel(y_truth), label='truth')
        plt.xlabel('time (s)', fontsize=18)
        plt.ylabel('R (m)', fontsize=18)
        plt.legend(fontsize=16)
        plt.title('GRF(l={:.1})'.format(i), fontsize=22)
        plt.savefig(Par['model'] + '_' + str(Par['m']) + '/predictions/'+str(i)+'.png')
        plt.close()

main()