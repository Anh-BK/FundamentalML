from __future__ import division, print_function, unicode_literals
import numpy as np 

class Perceptron:

    def __init__(self, eta = 0.1, epochs = 10):

        self.eta = eta # learning rate
        self.epochs = epochs # number of epochs
    # activation fuction: Sign() fuction
    def activationFunction(self, z):

        return np.sign(z)
    # To process original input data
    def inputData(self, X , y):

        ones = np.ones((X.shape[0], 1))
        self.Xbar = np.concatenate((ones, X), axis = 1)
        self.y = y
    #using SGD to optimize loss function
    def train(self):

        theta_init = np.zeros((3, 1))
        self.w = [theta_init]
        count = 1
        for _ in range(self.epochs):
            #shuffle data
            index = list(range(len(self.Xbar)))
            np.random.shuffle(index)    
            for sample in index:
                # if xi is misclassified w will be updated
                # if not do nothing the test with entire input data
                zi = np.dot(self.Xbar[sample], self.w[-1])
                if self.activationFunction(zi) != self.y[sample]:

                    w_new = self.w[-1] + self.eta * self.y[sample] * (self.Xbar[sample]).reshape(3, 1)
                    (self.w).append(w_new)
                    count += 1
                # test with entire inputdataset
                z = np.dot(self.Xbar, self.w[-1])
                if np.array_equal(self.activationFunction(z), self.y):

                    return self.w[-1], count

        return self.w[-1], count

    def predicted_data(self, inputdata):

        y_hat = self.activationFunction(np.dot(inputdata, self.w[-1]))
        
        return y_hat


import pandas as pd 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

PLA = Perceptron()
PLA.inputData(X, y)
w, count = PLA.train()
print(w)
print(count)
#visualization
import matplotlib.pyplot as plt

cluster_1 =[]
cluster_2 = []
for index in range(len(X)):

    if y[index] == -1 :

        cluster_1.append(X[index])

    else:

        cluster_2.append(X[index])

cluster_1 = np.array(cluster_1)
cluster_2 = np.array(cluster_2)

plt.scatter(cluster_1[:,0], cluster_1[:,1], marker='+')
plt.scatter(cluster_2[:,0], cluster_2[:,1], marker='o')

i = np.linspace(np.amin(X[:,:1]), np.amax(X[:,:1]))
slope = -(w[1]/w[2])
intercept = -(w[0]/w[2])
#d = slope * i + intercept
d = (slope * i) + intercept
plt.plot(i, d)
plt.title("Perceptron")
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

