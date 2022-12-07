# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#---------Intro-----------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)


node1 = tf.constant([1,2,3,4,5], name='node1')
node2 = tf.constant([1,2,3,4,5], name='node2')

node3 = node1*node2
node3.numpy()

node4 = tf.reduce_sum(node3)

#---------Linear regression-----------------

class MyModel:
    def __init__(self,w,b):
        self.w = w
        self.b = b
    
    def __call__(self,x):
        return x * self.w +self.b
    
    def loss_calc(self,predicted_y,target_y):
        loss = tf.reduce_mean((predicted_y-target_y)**2)
        return loss
    
    def train(self,dataset,learning_rate,epochs):
        wb_list = []
        for i in range(epochs):
            for x,y in dataset:
                with tf.GradientTape() as tape:
                    y_pred = x['x'] * self.w +self.b
                    loss = self.loss_calc(y_pred,y)

                [dl_dw, dl_db] =tape.gradient(loss, [self.w,self.b])
                dl_dw = tf.reduce_mean(dl_dw)
                dl_db = tf.reduce_mean(dl_db)
                self.w.assign_sub(tf.reshape(learning_rate*dl_dw,(1,)))
                self.b.assign_sub(tf.reshape(learning_rate*dl_db,(1,)))

            wb_list.append([self.w.numpy(), self.b.numpy()])
        print('Final w is:',self.w,'\nFinal b is:',self.b,'\nLoss is:',loss)
        return wb_list

if __name__ == '__main__':
    # batch_size = 32
    # lr_df = pd.read_csv(
    #     r'G:\My Drive\Primerose 18\Tensorflow\data_for_linear_regression_tf.csv').astype('float32')
    # dataset = tf.data.Dataset.from_tensor_slices((lr_df['x'],lr_df['y']),'XY')
    dataset = tf.data.experimental.make_csv_dataset(
        r'G:\My Drive\Primerose 18\Tensorflow\data_for_linear_regression_tf.csv',
        batch_size=32,
        column_names=['x','y'],
        label_name='y',
        num_epochs=1,
        shuffle=False)

    w1 = tf.Variable(tf.random.normal((1,), name='weights'))
    b1 = tf.Variable(tf.zeros(1, dtype=tf.float32, name='bias'))


    model1 = MyModel(w1, b1)
    wb1 = model1.train(dataset,0.1,100)

    plt.subplot()
    plt.title('Weights and Biases')
    plt.plot(np.asarray(wb1)[:,0], label='Weights')
    plt.axhline(y=-1, linestyle="--" ,  label='Real weight')
    plt.axhline(y=4, linestyle="--" ,color='orange', label='Real bias')
    plt.plot(np.asarray(wb1)[:,1], label='Biases')
    plt.legend()





        
            
        


