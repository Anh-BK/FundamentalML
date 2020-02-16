import numpy as np
import pandas as pd 
import scipy

def softmaxFunction(Z):
    
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)

    return A

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

iris = pd.DataFrame({'SepalLengthCm': df.iloc[:,0],
                     'SepalWidthCm': df.iloc[:,1],
                     'PetalLengthCm': df.iloc[:,2],
                     'PetalWidthCm': df.iloc[:,3],
                     'Species': df.iloc[:,4]})
iris.loc[-1] = [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
iris.index = iris.index + 1
iris = iris.sort_index()

np.random.seed(2)
random_index = np.random.rand(len(iris)) < 0.8
train = iris[random_index].reset_index()
test = iris[~random_index].reset_index()
train = train.drop(['index'], axis= 1)
test = test.drop(['index'], axis = 1)

X = (train.iloc[:,0:4].values).T
ones = np.ones((1, X.shape[1]))
Xbar = np.concatenate((ones, X), axis = 0)
N = Xbar.shape[1]
d = Xbar.shape[0]
y = train['Species']
kind_of_label = y.unique()
y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
one_hot_coding = np.array(pd.get_dummies(y)).T
yi = one_hot_coding[:,0]
ai = np.array([[0.1, 0.9, 0]])
e = ai - yi
xi = Xbar[:,0].reshape(d, 1)
W = np.ones((d, 3))
Z = np.dot(W.T, Xbar)
A = softmaxFunction(Z)
print(e.T)
print(Xbar[:,0].shape)
print(A)
print(Z.shape)
