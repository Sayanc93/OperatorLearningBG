from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from seq import Seq_Model


def preprocess(dataset, m, npoints_output, is_test=False):
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    # print('p shape:', p.shape)
    # print('t shape:', t.shape)
    # print('r shape:', r.shape)

    # print('p:', p[0, :])
    # print('t:', t[0])
    # print('r:', r[0, :])

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m))  # [1500, m]
    # X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y = R(np.linspace(t_min, t_max, m))  # [1500, npoints_output]

    if is_test:
        X_func = X_func[:50]
        y = y[:50]

    # print('X_func shape:', X_func.shape)
    # print('y shape:', y.shape)

    return X_func, y


def normalize(X_func, y, Par):
    X_func = (X_func - Par['p_mean'])/Par['p_std']
    y = (y - Par['r_mean'])/Par['r_std']

    return X_func.astype(np.float32), y.astype(np.float32)


@tf.function()
def train(seq_model, X, y):
    with tf.GradientTape() as tape:
        y_hat = seq_model(X)
        loss = seq_model.loss(y_hat, y)[0]

    gradients = tape.gradient(loss, seq_model.trainable_variables)
    seq_model.optimizer.apply_gradients(
        zip(gradients, seq_model.trainable_variables))
    return(loss)


def main():
    Par = {}
    Par['address'] = 'seq'

    train_dataset = np.load('../data/0.1/res_1000.npz')
    test_dataset = np.load('../data/0.08/res_1000.npz')

    m = 20
    npoints_output = 200

    X_train, y_train = preprocess(
        train_dataset, m, npoints_output)
    X_test, y_test = preprocess(
        test_dataset, m, npoints_output, is_test=True)

    X_train = np.reshape(X_train, [-1, m, 1])
    X_test = np.reshape(X_test, [-1, m, 1])
    y_train = np.reshape(y_train, [-1, m, 1])
    y_test = np.reshape(y_test, [-1, m, 1])

    Par['p_mean'] = np.mean(X_train)
    Par['p_std'] = np.std(X_train)

    Par['r_mean'] = np.mean(y_train)
    Par['r_std'] = np.std(y_train)

    X_train, y_train = normalize(X_train, y_train, Par)
    X_test, y_test = normalize(X_test, y_test, Par)

    # print('X_func_train: ', X_train.shape, '\ny_train: ', y_train.shape)
    # print('X_func_test: ', X_test.shape, '\ny_test: ', y_test.shape)

    seq_model = Seq_Model(Par)
    n_epochs = 100000
    batch_size = 150

    print("Seq2Seq Training Begins")
    begin_time = time.time()

    for i in range(n_epochs+1):
        for end in np.arange(batch_size, X_train.shape[0]+1, batch_size):
            start = end - batch_size

            # print('Epoch: ', i, '\tBatch: ', start, '\t', end)

            loss = train(
                seq_model, X_train[start:end], y_train[start:end])

            # print('Loss: ', loss)

        if i % 1000 == 0:
            seq_model.save_weights(Par['address'] + "/model_"+str(i))

            train_loss = loss.numpy()

            y_hat = seq_model(X_test)

            val_loss = np.mean((y_hat - y_test)**2)

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(
                val_loss) + ", elapsed time: " + str(int(time.time()-begin_time)) + "s")

            seq_model.index_list.append(i)
            seq_model.train_loss_list.append(train_loss)
            seq_model.val_loss_list.append(val_loss)

    # Convergence plot
    index_list = seq_model.index_list
    train_loss_list = seq_model.train_loss_list
    val_loss_list = seq_model.val_loss_list

    plt.plot(index_list, train_loss_list, label="train")
    plt.plot(index_list, val_loss_list, label="val")
    plt.legend()
    plt.title('Seq2Seq', fontsize=22)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(Par["address"] + "/convergence.png")

    print('Complete')


main()
