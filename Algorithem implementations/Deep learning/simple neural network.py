# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:19:20 2022

@author: Ben
"""

import numpy as np
import pandas as pd

def Sigmoid(x,derivative = False ):
    if derivative == False:
        return 1/(1+np.exp(-x))
    else:
        return x * (1-x)
  
def ReLU(x,derivative = False):
    if derivative == False:
        return x * (x > 0)
    else:
        x[x<=0] = 0
        x[x>0] = 1
        return x

#V1 - single layer(input only)

x1 = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y1 = np.array([[0],[0],[1],[1]])
epochs1 = 1000
init_weights1 = np.random.randn(3,1)


def iterate_v1(X,Y,init_weights,epochs):
    for i in range(epochs):
        l1 = Sigmoid(np.dot(X,init_weights))
        l1_error = l1 - Y  
        l1_delta = l1_error * Sigmoid(l1,True)
        init_weights -= np.dot(X.T,l1_delta)
    print(l1)
        
iterate_v1(x1,y1, init_weights1,epochs1)

#V2 - 2 layer(1 hidden)

x2 = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y2 = np.array([[0],[1],[1],[0]])
epochs2 = 10000
l1_weights2 = np.random.randn(3,4)
l2_weights2 = np.random.randn(4,1)
b1 = np.random.randn(4,1)
b2 = np.random.randn(1,1)


def iterate_v2(X,Y,l1_weights,l2_weights,epochs):
    global b1,b2
    for i in range(epochs):
        l1 = Sigmoid(np.dot(X,l1_weights) + np.dot(X,np.square(l1_weights)) + b1.T)
        l2 = Sigmoid(np.dot(l1,l2_weights) + b2.T)
        l2_error = l2 - Y
        l2_delta = l2_error * Sigmoid(l2,True)
        l1_error = l2_delta.dot(l2_weights.T)
        l1_delta = l1_error * Sigmoid(l1,True)
        l1_weights -= np.dot(X.T,l1_delta)
        l2_weights -= np.dot(l1.T,l2_delta)
        b1 = b1 - np.sum(l1_delta,axis=1,keepdims=True)
        b2 = b2 - np.sum(l2_delta,axis=1)
    print(l2,"\n",l1_weights,"\n",l2_weights)
     
iterate_v2(x2,y2,l1_weights2,l2_weights2,epochs2)

#V3 attempt





def iterate_v3(X,Y,neural_size,epochs,sigma):
    """
    This function recieves inputs,observed outputs and employees a neural
    network with m layers and number of neurons in each layer

    Parameters
    ----------
    X : np.array (n,m)
        An array with size n for size of input dataset and each
        elemnt m is a vector of the inputs 
    Y : np.array (n,m)
        An array with size n for size of observed dataset and each
        elemnt m is a vector of the oberved results
    neural_size : int list
        A list with size of number of layers where each element is
        number of neurons in each layer
    epochs : int
        Number of iterations to perform on the network
    sigma : function
        Which activation function to use

    Returns
    -------
    neuron_value : list
        A list containing matrices where each matrix is the neuron values
        calculated for each layer over the entire dataset
    weights_list : list
        A list containing matrices where each matrix is the weight values
        calculated for each weight connecting the neurons over diffrent layers

    """
    weights_list = []
    neuron_value = list(range(len(neural_size)))
    neuron_value[0] = X
    for i in range(len(neural_size)-1): #initialize the weights with normalized random values
        weights_list.append(np.random.randn(neural_size[i],neural_size[i+1]))
        
    layer_err = list(range(len(weights_list))) 
    layer_delta = list(range(len(weights_list)))
    
    for i in range(epochs): #NN training 
        for i,y in enumerate(weights_list): #Forward propogation
            neuron_value[i+1] = sigma(np.dot(neuron_value[i],y))
        for i in range(len(weights_list),0,-1): # Backward propogation
            if i == len(layer_err):
                layer_err[i-1] = (neuron_value[i] - Y)
            else:
                layer_err[i-1] = layer_delta[i].dot(weights_list[i].T)
            layer_delta[i-1] = layer_err[i-1] * sigma(neuron_value[i],True)

        for i in range(len(weights_list)): #update the weights for the next epoch
            weights_list[i] -= np.dot(neuron_value[i].T,layer_delta[i])
            
    print(neuron_value[len(weights_list)])
    return neuron_value,weights_list
        
x3 = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y3 = np.array([[1],[1],[1],[0]])
epochs3 = 10000
neural_size3 = [3,4,1]
neuron_results, weights = iterate_v3(x3,y3,neural_size3,epochs3,Sigmoid)
