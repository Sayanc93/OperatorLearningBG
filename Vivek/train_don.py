import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from don import DeepONet_Model

def preprocess(dataset, m, npoints_output, is_test=False):
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m)) #[1500, m]
    X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y      = R(np.ravel(X_loc)) #[1500, npoints_output]

    if is_test:
        X_func = X_func[:50]
        y = y[:50]

    return X_func, X_loc, y

def normalize(X_func, X_loc, y, Par):
    X_func = (X_func - Par['p_mean'])/Par['p_std']
    X_loc  = (X_loc - np.min(X_loc))/(np.max(X_loc) - np.min(X_loc))
    y = (y - Par['r_mean'])/Par['r_std']

    return X_func.astype(np.float32), X_loc.astype(np.float32), y.astype(np.float32)

@tf.function(jit_compile=True)
def train(don_model, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.loss(y_hat, y)[0]

    gradients = tape.gradient(loss, don_model.trainable_variables)
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return(loss)

def main():
    Par = {}
    Par['address'] = 'don'

    train_dataset = np.load('../data/0.1/res_1000.npz')
    test_dataset = np.load('../data/0.08/res_1000.npz')

    m = 20
    npoints_output = 200

    X_func_train, X_loc_train, y_train = preprocess(train_dataset, m, npoints_output)
    X_func_test, X_loc_test, y_test = preprocess(test_dataset, m, npoints_output, is_test=True)

    Par['p_mean'] = np.mean(X_func_train)
    Par['p_std']  = np.std(X_func_train)

    Par['r_mean'] = np.mean(y_train)
    Par['r_std']  = np.std(y_train)

    X_func_train, X_loc_train, y_train = normalize(X_func_train, X_loc_train, y_train, Par)
    X_func_test, X_loc_test, y_test    = normalize(X_func_test, X_loc_test, y_test, Par)

    print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)


    don_model = DeepONet_Model(Par)
    n_epochs = 100000
    batch_size = 150

    print("DeepONet Training Begins")
    begin_time = time.time()


    for i in range(n_epochs+1):
        for end in np.arange(batch_size, X_func_train.shape[0]+1, batch_size):
            start = end - batch_size
            loss = train(don_model, X_func_train[start:end], X_loc_train, y_train[start:end])

        if i%1000 == 0:

            don_model.save_weights(Par['address'] + "/model_"+str(i))

            train_loss = loss.numpy()

            y_hat = don_model(X_func_test, X_loc_test)

            val_loss = np.mean( (y_hat - y_test)**2 )

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)

    #Convergence plot
    index_list = don_model.index_list
    train_loss_list = don_model.train_loss_list
    val_loss_list = don_model.val_loss_list

    plt.plot(index_list, train_loss_list, label="train")
    plt.plot(index_list, val_loss_list, label="val")
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(Par["address"] + "/convergence.png")

    print('Complete')


main()
