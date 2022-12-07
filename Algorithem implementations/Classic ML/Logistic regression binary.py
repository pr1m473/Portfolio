import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




class Logistic_regression:
    def __init__(self,input,label):
        self.X = input
        self.Y = label.reshape(-1,1)
        #Weights are of size of input, output size is currently only 1!
        self.W = np.random.normal(size=(input[0].size,label[0].size))
        self.W = np.zeros((input[0].size,label[0].size))
        # self.W = np.ones((input[0].size,label[0].size))
        # self.b = np.random.normal(size=(label[0].size))
        # self.b = np.zeros((label[0].size))
        # self.b = np.ones((label[0].size))
        self._split(shuffle=True)

    def _split(self, test_size=0.2, random_seed=None, shuffle=False):
        '''
        performs train_test_split on the data and labels

        :param test_size: float, [0,1], optional:the ratio to split the test data. default is 0.2
        :param random_seed: float, optional:state a random seed for reproducible results, default is None
        :param shuffle: bool, optional:whether to shuffle the data, default is None
        :return: None
        '''
        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(self.X, self.Y,
                             test_size=test_size, random_state=random_seed, shuffle=shuffle)


    def _sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def _predict(self, x):
        self.predict = []
        for obj in x:
            if obj > 0.5:
                self.predict.append(1)
            else:
                self.predict.append(0)
        return np.array(self.predict)

    def _negative_log(self,h_x,y):
        return -(y.T@np.log(h_x)+(1-y).T@np.log(1-h_x))/len(self.Y)

    def _calc_gradient(self,data,error):
        return data.T@error


    def train(self,epochs=100,lr=0.1):
        losses = []
        for epoch in range(epochs):
            y_pred = self._sigmoid(self.train_data@self.W+self.b)
            loss = self._negative_log(y_pred,self.train_label)
            losses.append(loss)
            error = y_pred - self.train_label
            grad = self._calc_gradient(self.train_data,error)

            self.W -= lr*grad
            self.b -= lr*np.mean(error)

    def _calc_accuracy(self,y_pred,y_label,title=''):
        accuracy = 0
        prediction = self._predict(y_pred)
        for i in range(len(prediction)):
            accuracy += prediction[i] == y_label[i]
        accuracy = accuracy / len(y_pred) * 100
        print(f'The accuracy of the {title} is: {np.round(accuracy[0],2)}%')

    def evaluate(self):
        self._calc_accuracy((self._sigmoid(self.train_data@self.W+self.b)),self.train_label,'train set')
        self._calc_accuracy((self._sigmoid(self.test_data @ self.W + self.b)), self.test_label, 'test set')



if __name__ == '__main__':
    iris_db = datasets.load_iris()
    dataset_df = pd.DataFrame(np.hstack([iris_db.data, iris_db.target.reshape((-1, 1))]))
    dataset_df[5] = [iris_db.target_names[int(x)] for x in dataset_df[4]]
    dataset_df = dataset_df.rename(columns={4: 'Label'})
    dataset_df['Label'] = [1 if dataset_df[5][i] == 'setosa' else 0 for i, x in enumerate(dataset_df['Label'])]
    X_data = np.array(dataset_df.iloc[:, :2])

    plt.figure()
    plt.scatter(X_data[dataset_df['Label'] == 1][:, 0], X_data[dataset_df['Label'] == 1][:, 1], c='blue')
    plt.scatter(X_data[dataset_df['Label'] == 0][:, 0], X_data[dataset_df['Label'] == 0][:, 1], c='red')
    plt.show()

    iris_lg = Logistic_regression(X_data,np.array(dataset_df['Label']))
    iris_lg.train(10000,0.02)
    iris_lg.evaluate()



