from __future__ import division, print_function, unicode_literals
import numpy as np 
import pandas as pd

class SoftmaxRegression:

    def __init__(self, eta = 0.1, epochs = 10):

        self.eta = eta
        self.epochs = epochs

    def softmaxFunction(self, Z):
        
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        A = e_Z/e_Z.sum(axis = 0)

        return A  
        
    def inputData(self, X, Y):
        #pre-proccessing original data for trainning 
        self.N = X.shape[1] # numbers of samples
        self.d = X.shape[0] # dimension of sample
        self.C = len(Y.unique()) # numbers of classes
        ones = np.ones((1, self.N))
        self.Xbar = np.concatenate((ones, X), axis = 0)
        self.one_hot_coding = np.array(pd.get_dummies(Y)).T


    def train(self):
        
        count = 1
        W_init = np.zeros((self.d + 1, self.C)) # shape of W that is (d+1)xC (C is number of classes)
        self.W = [W_init]
        for _ in range(self.epochs):
            
            index = list(range(self.N))
            np.random.shuffle(index)
            for sample in index:

                count += 1
                Z = np.dot((self.W[-1]).T, self.Xbar)
                A = self.softmaxFunction(Z)
                ai = A[:,sample].reshape(self.C, 1)
                xi = (self.Xbar[:,sample]).reshape(self.d + 1, 1)
                yi = (self.one_hot_coding[:,sample]).reshape(self.C, 1) #label of sample that is represented in one hot coding
                ei = ai - yi
                W_new = self.W[-1] - self.eta * np.dot(xi, ei.T)
                (self.W).append(W_new)
                if count % 5 == 0:

                    if np.linalg.norm(W_new - self.W[-5]) < 1e-3:

                        return self.W[-1]
        
        return self.W[-1]

    def predictedOutput(self, test):

        ones = np.ones((1, test.shape[1]))
        Xbar = np.concatenate((ones, test), axis = 0) 
        Z = np.dot((self.W[-1]).T, Xbar)
        A = self.softmaxFunction(Z)
        prediction = []
        for i in range(A.shape[1]):

            prediction.append(np.where(A == A[:,i].max())[0])
        
        prediction = np.array(prediction)

        return prediction

#dataframe
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
#repair dataframe
iris = pd.DataFrame({'SepalLengthCm': df.iloc[:,0],
                     'SepalWidthCm': df.iloc[:,1],
                     'PetalLengthCm': df.iloc[:,2],
                     'PetalWidthCm': df.iloc[:,3],
                     'Species': df.iloc[:,4]})
iris.loc[-1] = [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
iris.index = iris.index + 1
iris = iris.sort_index()
# separate dataframe become two datasets: train and test
np.random.seed(2)
random_index = np.random.rand(len(iris)) < 0.6
train = iris[random_index].reset_index()
test = iris[~random_index].reset_index()
train = train.drop(['index'], axis= 1)
test = test.drop(['index'], axis = 1)
#trainning dataset
X_train = (train.iloc[:,0:4].values).T #dataset for trainning
Y_train = train['Species']
Y_train = Y_train.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}) #label of dataset
#testing set
X_test = (test.iloc[:,0:4].values).T
Y_test = test['Species']
Y_test = Y_test.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
Y_test = np.array(Y_test)
#run
MultipleClassifier = SoftmaxRegression(0.01, 100)
MultipleClassifier.inputData(X_train, Y_train)
W = MultipleClassifier.train()
print(W)
#result of comparetion
prediction = MultipleClassifier.predictedOutput(X_test)
correct = np.sum(Y_test == prediction.T)
accuracy = (correct/len(Y_test))*100
print(prediction.T)
print(Y_test)
print(accuracy)