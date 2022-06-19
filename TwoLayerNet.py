import numpy as np
import matplotlib.pyplot as plt
import util
import layer
import nn
import optimizer
import pickle #實現 Python 物件的存儲及恢復
import loss 
from PIL import Image
import random
import time

X_train = np.load("/home/ma/Tensorflow_Lenet5/X_train_G.npy")
Y_train = np.load("/home/ma/Tensorflow_Lenet5/Y_train_G.npy")
X_val = np.load("/home/ma/Tensorflow_Lenet5/X_val_G.npy")
Y_val = np.load("/home/ma/Tensorflow_Lenet5/Y_val_G.npy")
X_test = np.load("/home/ma/Tensorflow_Lenet5/X_test_G.npy")
Y_test = np.load("/home/ma/Tensorflow_Lenet5/Y_test_G.npy")


X_train, X_val,X_test = X_train/float(255), X_val/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_val -= np.mean(X_val)
X_test -= np.mean(X_test)

X_train = X_train.reshape(63325,1024)
X_val = X_val.reshape(450,1024)
X_test = X_test.reshape(450,1024)

batch_size = 1000
D_in = 1024
D_out = 50

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
H=500
model = nn.TwoLayerNet(batch_size, D_in, H, D_out)

losses = []
train_acc = []
val_acc = []
test_acc = []
#optim = optimizer.SGD(model.get_params(), lr=0.000001, reg=0)
optim = optimizer.SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
criterion = loss.CrossEntropyLoss()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.sample(range(N),batch_size)
    X_batch = []
    Y_batch = []
    for inx in i:
        X_batch.append(X[inx])
        Y_batch.append(Y[inx])
    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    return X_batch, Y_batch

# TRAIN
ITER = 3000
start = time.time()
for i in range(ITER):
    # get batch, make onehot
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
    Y = Y_batch
    Y_batch = util.MakeOneHot(Y_batch, D_out)
    
    # forward, loss, backward, step
    Y_pred = model.forward(X_batch)
    result = np.argmax(Y_pred, axis=1) - Y
    result = list(result)
    acc_train = result.count(0)/1000
    train_acc.append(acc_train)
    loss, dout = criterion.get(Y_pred, Y_batch)
    losses.append(loss)
    model.backward(dout)
    optim.step()

    # val
    Y_pred_val = model.forward(X_val)
    result = np.argmax(Y_pred_val, axis=1) - Y_val
    result = list(result)
    acc_val = result.count(0)/450
    val_acc.append(acc_val)
    print("%s%% iter: %s, loss: %s, acc: %s" % (100*i/ITER, i, loss, acc_val))
    
    # test
    Y_pred_test = model.forward(X_test)
    result = np.argmax(Y_pred_test, axis=1) - Y_test
    result = list(result)
    acc_test = result.count(0)/450
    test_acc.append(acc_test)
   # acclist.append(acc)

end = time.time()
print(format(end-start))
# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

#with open("losses.pkl","wb") as f:
#    pickle.dump(losses, f)
    
#with open("acc.pkl","wb") as f:
#    pickle.dump(acclist, f)

util.draw_acc(train_acc,val_acc,test_acc,'acc')
util.draw(losses,'loss')