#declare libary
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import random 
#initial data
np.random.seed(2) #not actually neccersary
X = np.random.rand(1000,1)
y = 4 + 3*X + 0.2*np.random.randn(1000,1)
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)
#gradient of loss fuction with a single point (xi, yi) 
def grad(theta_old, sample):

    xi = Xbar[sample,:] 
    yi = y[sample]

    return (xi * (np.dot(xi, theta_old) - yi)).reshape(2,1)
#SGD
def SGD(theta_init):

    w = [theta_init]
    count = 0
    loss = []
    theta_new = np.zeros((2, 1))
    for _ in range(10):
        #Initially, data will be shuffled.
        index = list(range(len(Xbar)))
        np.random.shuffle(index) 
        for sample in index :
            
            count += 1 # times of update
            theta_new = w[-1] - 0.1 * grad(w[-1], sample) # learning rate = 0.9
            loss.append(.5 * (np.linalg.norm(y - np.dot(Xbar, w[-1]),2) ** 2)/len(Xbar)) 
            w.append(theta_new)
            #check
            if count % 10 == 0:

                if np.linalg.norm(theta_new - w[-10])/len(theta_new) < 1e-3:

                    return w, count, loss
            
    return w, count, loss    

theta_init = np.zeros((2, 1))
w, count, loss = SGD(theta_init)
times = np.array(range(count))
loss = np.array(loss)
print(len(loss))
print(w[-1])
plt.plot(times[0:99], loss[0:99]) #for first 100 times
plt.axis = ([times.min(),times.max(),0,loss.max()])
plt.xlabel('Times')
plt.ylabel('Loss')
plt.title('SGD')
plt.show()









