from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt 

# input data
np.random.seed(2) #not actually neccersary
X = np.random.rand(1000,1)
y = 4 + 3*X + 0.2*np.random.randn(1000,1)
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)
# gradient
def grad(theta):

    return np.dot(Xbar.T, (np.dot(Xbar, theta) - y))/len(Xbar)
# momentum
def Momentum(theta_init):

    w = [theta_init]
    v_old = np.zeros((2, 1))
    loss_init = .5 * ((np.linalg.norm(np.dot(Xbar,w[-1]) - y))**2)/len(Xbar)
    loss = [loss_init]
    iterations = 1
    while True:

        v_new = 0.9 * v_old + 0.9 * grad(w[-1])
        theta_new = w[-1] - v_new
        w.append(theta_new)
        loss.append(.5*(np.linalg.norm(y - np.dot(Xbar, w[-1]))**2)/len(Xbar))
        iterations += 1
        if np.linalg.norm(grad(theta_new)) < 1e-3:

            return w, loss, iterations 

theta_init = np.zeros((2, 1))
(w, loss, iterations) = Momentum(theta_init)
iterations = np.array(range(iterations))
loss = np.array(loss)
print("Iterations: ",len(iterations))
plt.plot(iterations, loss)
plt.axis([iterations.min(), iterations.max(), 0, loss.max()])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Momentum")
plt.show()


