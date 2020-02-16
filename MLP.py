from __future__ import division, print_function, unicode_literals
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

class MLP:

    def __init__(self, eta, epochs)

        self.eta = eta
        self.epochs = epochs
    
    def softmax(self, Z):

        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        A = e_Z/e_Z.sum(axis = 0)

        return A
    # gradient of relu
    def reluDerivative(self,X):

        X[X <= 0] = 0
        X[X > 0] = 1

        return X
    #Inputdata
    def inputData(self, X, y):

        self.d = X.shape[0]
        self.N = X.shape[1]
        ones = np.ones((1, self.N))
        self.Xbar = np.concatenate((ones, X), axis = 0)
        y = pd.Series(y)
        y = y.unique()
        self.Y =np.array(pd.get_dummies(y)).T  # y in one hot coding
    #Feedforward
    def forward(self):

        Z1 = np.dot(W1.T, self.Xbar)
        self.A1 = np.maximum(0, Z1)
        Z2 = np.dot(W2.T, A1)
        self.Y_hat = self.softmax(Z2)
    #backward pass
    def backward(self):

        E2 = (self.Y_hat - self.Y)/self.N
        dJ_dW2 = np.dot(self.A1, E2.T)
        dJ_db2 = E2.sum(axis = 1)
        E1 = 





