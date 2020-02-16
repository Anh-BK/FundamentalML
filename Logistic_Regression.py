from __future__ import division, print_function, unicode_literals
import numpy as np

class Neural:
    def __init__(self, eta = 0.1, epochs = 10):

        self.eta = eta
        self.epochs = epochs
    # sigmoid function
    def sigmoid(self, s):

        return 1/(1 + np.exp(-s))
    # proccesing original inputdata    
    def inputData(self, X, y):

        ones = np.ones((len(X), 1))
        self.Xbar = np.concatenate((ones, X), axis = 1)
        self.y = y
    # using SGD algorithm
    def train(self):

        w_init = np.zeros((3, 1))
        self.w = [w_init]
        count = 1 # counting number of updating
        for _ in range(self.epochs):

            index = list(range(len(self.Xbar))) # index of a sample in the dataset
            np.random.shuffle(index) # shuffle index to pick a radom sample in the dataset
            for sample in index:
                
                si = np.dot(self.Xbar[sample], self.w[-1])
                w_new = self.w[-1] + self.eta * (self.y[sample] - self.sigmoid(si)) * (self.Xbar[sample]).reshape(3, 1)
                count += 1
                (self.w).append(w_new)
                if count % 10 == 0:
                    
                    if np.linalg.norm(w_new - self.w[-10]) < 1e-3 :

                        return self.w[-1]
            
            return self.w[-1]
    
    def output(self, s):
        
        z = np.dot(s, self.w[-1])

        return self.sigmoid(z)

import pandas as pd 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0,2]].values

Logistic_Regression = Neural()
Logistic_Regression.inputData(X, y)
w = Logistic_Regression.train()
print(w)

#visualiztion
import matplotlib.pyplot as plt

cluster1 = []
cluster2 = []
for i in range(len(X)):

    if y[i] == 1:

        cluster1.append(X[i])
    
    else:
        cluster2.append(X[i])

cluster1 = np.array(cluster1)
cluster2 = np.array(cluster2)        

#make data
xm = np.arange(-1,8 , 0.025)
xlen = len(xm)
ym = np.arange(0,10, 0.025)
ylen = len(ym)
xm, ym = np.meshgrid(xm, ym)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]

sm = w_0 + w_1 * xm + w_2 * ym 
zm = Logistic_Regression.sigmoid(sm)
CS = plt.contourf(xm, ym, zm, 200, cmap = 'jet')

plt.plot(cluster1[:,0], cluster1[:,1],'rs' , markersize = 8, alpha = 1)
plt.plot(cluster2[:,0], cluster2[:,1],'bo' , markersize = 8, alpha = 1)

plt.axis('equal')
plt.xlim(0,np.amax(X[:,0]))
plt.ylim(0,np.amax(X[:,1]))
#hide ticks
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlabel('$x_1$', fontsize = 20)
plt.ylabel('$x_2$', fontsize = 20)
plt.show()