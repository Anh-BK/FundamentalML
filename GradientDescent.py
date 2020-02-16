from __future__ import division, print_function, unicode_literals
import numpy as np 
# gradient of fuction 
def grad(x):
    return 2*x + 5*np.cos(x)
# Fuction 
def cost(x):
    return x**2 + 5*np.sin(x)
# Function of alogrithm  
def idea_GD(rate, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - rate*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)

    return x, it
# Test
(x1, it) = idea_GD(.1, 5)
print("Minimal point: %f, iterial: %d" %(x1[-1], it))
