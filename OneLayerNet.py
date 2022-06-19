import numpy as np
import matplotlib.pyplot as plt
import util
from PIL import Image
import random
from sklearn.metrics import accuracy_score
from urllib.request import urlopen
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

def training(X_train, y_train):
    dotprod=np.ndarray(50,dtype=np.float32)
    weights=np.zeros((50,X_train.shape[1]), dtype=np.float32)
    iterations=3000                           # Number of epochs
    alpha=0.0001                              # Learning Rate
    accu = []
    co=[]
    for iter in range(iterations) : 
        count=0
        for i in range(X_train.shape[0]): # iterating through each feature vector
            for j in range(50):     # weight vectors equivalent to 10 class labels 
                dotprod[j] = np.dot(X_train[i],weights[j])

            ind=y_train[i]                 # true class label
            maxi=np.argmax(dotprod,axis=None)
            y_pred=maxi             # got predicted corresponding class label
            if(y_pred.any() !=y_train[i].any()):
                weights[maxi] -= alpha*X_train[i]
                weights[ind] += alpha*X_train[i]
            elif(y_pred==y_train[i]):
                count=count+1
        accuracy = round(((count/y_train.shape[0])*100), 2)
        print(print("iter: %s, acc: %s" % (iter, accuracy)))
        accu.append(accuracy)
        co.append(count)
    util.draw(accu,'acc_One')
    return weights
    
def testing(X_test,w):
    
    ypred=[]
    activ=[]
    activ=np.dot(X_test,w.T)     #calculating activation
    
    for i in range(activ.shape[0]):
        ypred.append(np.argmax(activ[i]))

    return ypred           #predicted class label
def add_ones(X):
  return np.hstack((np.ones((X.shape[0], 1),dtype=int), X)) 

start = time.time()
w=training(X_train, Y_train)  #got updated weights
end = time.time()
print("time:",format(end-start))
predictions_for_Training=testing(X_train,w)
count=0
for i in range(Y_train.shape[0]):
    if predictions_for_Training[i]==Y_train[i]:
        count=count+1
print("Number of samples correctly classified for training data : ", count)
print('Accuracy in % :', accuracy_score(Y_train,predictions_for_Training)*100)

predictions_for_Training=testing(X_val,w)
count=0
for i in range(Y_val.shape[0]):
    if predictions_for_Training[i]==Y_val[i]:
        count=count+1
print("Number of samples correctly classified for val data : ", count)
print('Accuracy in % :', accuracy_score(Y_val,predictions_for_Training)*100)

predictions_for_Training=testing(X_test,w)
count=0
for i in range(Y_test.shape[0]):
    if predictions_for_Training[i]==Y_test[i]:
        count=count+1
print("Number of samples correctly classified for test data : ", count)
print('Accuracy in % :', accuracy_score(Y_test,predictions_for_Training)*100)