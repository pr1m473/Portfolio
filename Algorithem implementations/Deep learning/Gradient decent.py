# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:34:42 2022

@author: Ben
"""
import numpy as np
import matplotlib.pyplot as plt


def run_gradient_descent(X,y,start,learning_rate,epochs):
    t = start.copy()
    losses = []
    for epoch in range(epochs):
        y_predicted = X[:,0]*t[0]+X[:,1]*t[1]+X[:,2]*t[2]
        loss = ((y_predicted - y)**2).mean()
        losses.append(np.sqrt(loss))
        grad_t0 = ((y_predicted-y)*X[:,0]).mean()
        grad_t1 = ((y_predicted-y)*X[:,1]).mean()
        grad_t2 = ((y_predicted-y)*X[:,2]).mean()
        t[0] -= learning_rate*grad_t0
        t[1] -= learning_rate*grad_t1
        t[2] -= learning_rate*grad_t2
    return t,losses

def plot_losses(losses,title=''):
    plt.plot(losses)
    plt.title('Gradient Descent Loss '+title)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()




x = np.array([0,1,2],dtype=np.float32)
y = np.array([1,3,7],dtype=np.float32)
X = np.c_[np.ones_like(x),x,x**2]
start = np.array([2,2,0],dtype=np.float32)

#A


t1,losses1 = run_gradient_descent(X, y, start, 1, 100)
print(losses1)
plot_losses(losses1,"LR = 1")

t2,losses2 = run_gradient_descent(X, y, start, 0.1, 100)
print(losses2)
plot_losses(losses2,"LR = 0.1")

t3,losses3 = run_gradient_descent(X, y, start, 0.01, 400)
plot_losses(losses3,"LR = 0.01")

#B
"""
The only one that failed(LR=1), the LR was too big and so caused
to "Hop" over target minima
"""

#C

def run_gradient_decent_momentum(X,y,start,learning_rate,gamma,epochs):
    t = start.copy()
    losses = []
    v = np.array([0,0,0],dtype=np.float32)
    for epoch in range(epochs):
        y_predicted = X[:,0]*t[0]+X[:,1]*t[1]+X[:,2]*t[2]
        loss = ((y_predicted - y)**2).mean()
        losses.append(np.sqrt(loss))
        grad_t0 = ((y_predicted-y)*X[:,0]).mean()
        grad_t1 = ((y_predicted-y)*X[:,1]).mean()
        grad_t2 = ((y_predicted-y)*X[:,2]).mean()
        v[0] = gamma*v[0] + learning_rate*grad_t0
        v[1] = gamma*v[1] + learning_rate*grad_t1
        v[2] = gamma*v[2] + learning_rate*grad_t2
        t[0] -= v[0]
        t[1] -= v[1]
        t[2] -= v[2]
    return t,losses

t4, losses4 = run_gradient_decent_momentum(X, y, start, 0.1, 0.9, 400)
plot_losses(losses4,"LR = 0.1 with momentum 0.9")

def rmse_loss(y_predicted,y_true):
    return np.sqrt(((y_predicted-y_true)**2).mean())

def predict_vals(X,tetas):
    return X@tetas

def mse_loss_grad(y_predicted,y_true,X):
    return (y_predicted-y_true)@X/len(X)

