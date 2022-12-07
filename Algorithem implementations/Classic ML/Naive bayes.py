from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

class Naive_Bayes:
    '''
    Uses Naive base algorithm to make predictions
    '''
    def __init__(self,data_path):
        self.dataset = np.loadtxt(data_path,delimiter=',',skiprows=1)

    def split(self, test_size=0.2, random_seed=None, shuffle=False):
        '''
        performs train_test_split on the data and labels

        :param test_size: float, [0,1], optional:the ratio to split the test data. default is 0.2
        :param random_seed: float, optional:state a random seed for reproducible results, default is None
        :param shuffle: bool, optional:whether to shuffle the data, default is None
        :return: None
        '''
        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(self.dataset[:,:-1],self.dataset[:,-1],test_size=test_size,
                             random_state=random_seed, shuffle=shuffle)

    def train(self):
        '''
        calculates the train parameters: Mean,STD and P(class) for all the features and classes
        '''
        self.train_mean = np.zeros((len(np.unique(self.train_label)),self.train_data.shape[1]))
        self.train_std = np.zeros_like(self.train_mean)
        for i,cls in enumerate(np.unique(self.train_label)):
            self.train_mean[i] = np.mean(self.train_data[np.where(self.train_label == cls )],axis=0)
            self.train_std[i] = np.std(self.train_data[np.where(self.train_label == cls )],axis=0)
        self.train_pclass = np.unique(self.train_label,return_counts=True)[1].reshape((-1,1))/len(self.train_data)

    def calc_PDF(self,x,mean,std):
        '''
        Caculates P(x|C) using parameters from the train method for a single datapoint and feature

        :param x: float
        :param mean: float
        :param std: float
        :return: float
        '''
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2 / (2 * std ** 2)))

    def calc_pred(self,X):
        '''
        Calculates P(C|X) using all features for each class

        :param X: array: datapoint with all features
        :return: int: predicted label value
        '''
        pred_list = []
        for i in range(np.unique(self.test_label).shape[0]):
            pred_pcx = 1
            for j in range(self.test_data.shape[1]):
                pred_pcx *= self.calc_PDF(X[j],self.train_mean[i,j],self.train_std[i,j])
            pred_pcx *= self.train_pclass[i]
            pred_list.append(pred_pcx)
        return np.argmax(pred_list)

    def predict(self,test_data=None,test_label=None):
        '''
        Performs prediction on all test data and returns list of class predictions and accuracy metric
        External test dataset can be used

        :param test_data: array,optional: external test data. if None, instance test data is used
        :param test_label: array,optional: external test labels. if None, instance test label is used
        :return: Prints the resulting class prediction list and accuracy metric
        '''
        if test_data == None or test_label == None :
            test_data = self.test_data
            test_label = self.test_label
        self.result = [self.calc_pred(X) for X in test_data]
        self.accuracy = 0
        for i in range(len(self.result)):
            self.accuracy += self.result[i] == self.test_label[i]
        self.accuracy = self.accuracy / len(self.result) * 100
        print(f"The resulting classifications are {self.result}\n The accuracy is {np.round(self.accuracy,2)}%")

    def compare_GaussianNB(self):
        '''
        Runs the sklearn GaussianNB model and compares the predictions to Naive_Bayes class predictions

        :return: Prints the agreement percentage between the two models
        '''
        diabetes = GaussianNB()
        diabetes.fit(self.train_data, self.train_label)
        GaussianNB()
        compare_result = np.count_nonzero(diabetes.predict(self.test_data) == self.result)\
               / len(self.result) *100
        print(f"The resulting agreement between your model and sklearn GaussianNB model is {np.round(compare_result,2)}%")



if __name__ == '__main__':
    data_path = r'diabetes.csv'
    db_dataset = Naive_Bayes(data_path)
    db_dataset.split(0.2,42, True)
    db_dataset.train()
    db_dataset.predict()
    db_dataset.compare_GaussianNB()




