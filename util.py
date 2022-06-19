import numpy as np
import matplotlib.pyplot as plt
import random

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw(loss,title):
    t = np.arange(len(loss))
    plt.plot(t, loss)
    plt.savefig('plot_'+title+'.png')
    plt.show()

def draw_acc(train_acc,val_acc,test_acc,title):
    ITER = np.arange(len(train_acc))
    plt.plot(ITER, train_acc, color = 'blue', label='train')
    plt.plot(ITER, val_acc, color = 'orange', label='val')
    plt.plot(ITER, test_acc, color = 'green', label='test')
    plt.legend(loc = 'upper right')
    plt.savefig('plot_'+title+'.png')
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]
