import numpy as np
import pandas as pd 
import random 
#dataset
d0 = 2
d1 = h = 100 #size of hidden layer
d2 = C = 3 # number of classes
N = 100
eta = 0.1
X = np.zeros((d0, N*C)) #data matrix (each row = single example)
Y = np.zeros(N*C, dtype = 'uint8')
for j in range(C):

    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0,1,N) #radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
    X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    Y[ix] = j

Y = pd.Series(Y)
Y = np.array(pd.get_dummies(Y)).T

def network():

    model = dict(
        W1 = np.zeros((d0, d1)),
        b1 = np.zeros((d1, 1)),
        W2 = np.zeros((d1, d2)),
        b2 = np.zeros((d2, 1))
    )

    return model

def softmax(Z):

    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z/e_Z.sum(axis = 0)

    return A

def reluDerivative(X):

    X[X <= 0] = 0
    X[X > 0] = 1

    return X

X_train = X.T 
Y_train = Y.T

mini_X = X_train[50:100].T
mini_Y = Y_train[50:100].T
#feedforward
model = network()
Z1 = np.dot(model['W1'].T, mini_X) + model['b1']
A1 = np.maximum(Z1, 0)
Z2 = np.dot(model['W2'].T, A1) + model['b2']
mini_Yhat = softmax(Z2)
#backward
E2 = (mini_Yhat - mini_Y)/mini_Y.shape[1]
dJdW2 = np.dot(A1, E2.T)
dJdb2 = np.sum(E2, axis = 1, keepdims = True)
E1 = (np.dot(model['W2'], E2)) * reluDerivative(Z1)
dJdW1 = np.dot(mini_X, E1.T)
dJdb1 = np.sum(E1, axis = 1, keepdims = True)
#update weights
model['W2'] += -eta * dJdW2
model['b2'] += -eta * dJdb2
model['W1'] += -eta * dJdW1
model['b1'] += -eta * dJdb1
print(type(model))
print(Y.shape)
