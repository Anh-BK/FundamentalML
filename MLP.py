from __future__ import division, print_function, unicode_literals
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#model of network
d0 = 2
d1 = h = 100 #size of hidden layer
d2 = C = 3 # number of classes
eta = 0.9
def network():

    model = dict(
        W1 = 0.01*np.random.randn(d0, d1),
        b1 = 0.01*np.random.randn(d1, 1),
        W2 = 0.01*np.random.randn(d1, d2),
        b2 = 0.01*np.random.randn(d2, 1)
    )

    return model
#softmax function 
def softmax(Z):

    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z/e_Z.sum(axis = 0)

    return A
# gradient of relu function 
def reluDerivative(X):

    X[X <= 0] = 0
    X[X > 0] = 1

    return X
#test model
#dataset
N = 100 # numbeer of points per class
# dimentionality
# number of classes
X = np.zeros((d0, N*C)) #data matrix (each row = single example)
Y = np.zeros(N*C, dtype = 'uint8')
for j in range(C):

    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0,1,N) #radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
    X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    Y[ix] = j

Y = pd.Series(Y)
Y = np.array(pd.get_dummies(Y)).T #one hot coding 
#trainning
X_train = X.T 
Y_train = Y.T
count = 0
model = network()
loss = 10
while True:
    # shuffle data
    X_train = X.T 
    Y_train = Y.T

    X_train, Y_train = shuffle(X_train, Y_train)
    
    for i in range(0, X_train.shape[0], 50):

        count += 1
        mini_X = X_train[i:i+50].T
        mini_Y = Y_train[i:i+50].T
        # architechture of network
        
        #feedforward
        Z1 = np.dot(model['W1'].T, mini_X) + model['b1']
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(model['W2'].T, A1) + model['b2']
        mini_Yhat = softmax(Z2)
        #backward
        E2 = (mini_Yhat - mini_Y)/mini_Y.shape[1]
        dJdW2 = np.dot(A1, E2.T)
        dJdb2 = E2.sum(axis = 1, keepdims = True)
        E1 = (np.dot(model['W2'], E2)) * reluDerivative(Z1)
        dJdW1 = np.dot(mini_X, E1.T)
        dJdb1 = E1.sum(axis = 1, keepdims = True)
        #update weights
        model['W2'] += -eta * dJdW2
        model['b2'] += -eta * dJdb2
        model['W1'] += -eta * dJdW1
        model['b1'] += -eta * dJdb1
        #validation
        if count % 1000 == 0:

            Z1 = np.dot(model['W1'].T, X) + model['b1']
            A1 = np.maximum(Z1, 0)
            Z2 = np.dot(model['W2'].T, A1) + model['b2']
            Yhat = softmax(Z2)
            loss = -np.sum(Y * np.log(Yhat))/Y.shape[1]
            print('iter: %d, loss: %f' %(count, loss)) 

    if loss < 0.0129:

        break




    







