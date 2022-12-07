# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:34:42 2022

@author: Ben
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

class Binary_SVM:
    def __init__(self,X,Y,epochs,lr):
        self.label_dict = {'M':1,'B':-1}
        self.X = MinMaxScaler().fit_transform(X)
        self.Y = Y.replace((self.label_dict))
        self.epochs = epochs
        while lr > 1:
            lr = float(input('Learning rate must me smaller than 1!\n'
                             'enter a new number:'))
        self.lr = lr
        self.b = 1
        self.losses = []
        self.W = np.random.randn(self.X.shape[1],1)
        self.results = []


    def _hinge_loss(self,prediction):
        return np.maximum(np.array(1 - self.Y * prediction[:, -1]), np.zeros_like(self.Y)).mean()

    def _predict_vals(self,X):
        return X @ self.W + self.b

    def _SVM_loss_grad(self,prediction):
        grads = np.zeros_like(self.X)
        for i in range(len(self.Y)):
            if self.Y[i]*prediction[i] < 1:
                grads[i] = -self.Y[i]*self.X[i]
            else:
                grads[i] = np.zeros_like(self.Y[0])
        return grads.mean(axis=0)


    def plot_losses(self,title):
        plt.plot(self.losses)
        plt.title('Gradient Descent Loss '+title)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
        return

    def predict(self,test_data,test_label,pos_label,neg_label,compare=False):
        test_data = MinMaxScaler().fit_transform(test_data)
        test_label = test_label.reset_index()[1]
        predictions = self._predict_vals(test_data)
        accuracy = 0
        for x in predictions:
            if x >= 0:
                self.results.append(pos_label)
            else:
                self.results.append(neg_label)
        for i in range(len(predictions)):
            accuracy += self.results[i] == test_label[i]
        accuracy = accuracy / len(test_label) * 100

        print(f'The resulting predictions are:\n{self.results}')
        print(f'The calculated accuracy is: {np.round(accuracy,2)}%')

        if compare:
            test_label = test_label.replace((self.label_dict))
            sk_tree = SVC(kernel='linear')
            sk_tree.fit(self.X,self.Y)
            sk_score =  sk_tree.score(test_data,test_label)
            print(f'The sklearn score is: {sk_score}')

    def fit(self):
        for epoch in range(self.epochs):
            prediction = self._predict_vals(self.X)
            self.losses.append(self._hinge_loss(prediction))
            grad = self._SVM_loss_grad(prediction)
            self.W -= self.lr*grad.reshape(-1,1)
            self.b -= self.lr*np.sum(grad)


if __name__ == '__main__':
    data_df = pd.read_csv('G:\My Drive\Primerose 18\Decision trees\wdbc.data',header=None)
    train_data, test_data, train_label,test_label = train_test_split(data_df.iloc[:, 2:],
                                                       data_df.iloc[:, 1],
                                                       test_size=0.2, shuffle=False)
    diabetes_svm = Binary_SVM(train_data,train_label,200,0.5)
    diabetes_svm.fit()
    diabetes_svm.plot_losses('SVM')
    diabetes_svm.predict(test_data,test_label,'M','B',compare=True)



