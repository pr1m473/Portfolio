import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np


class Knn_ds:
    '''
    Introductory Class for performing K nearest neighbours analysis.
    first parameters required is the dataset data and dataset labels,as well as categories if required
    '''

    def __init__(self, dataset, labels, categories=None):
        self.dataset = dataset
        self.labels = labels
        self.categories = categories

    def split(self, test_size=0.2, random_seed=None, shuffle=False):
        '''
        performs train_test_split on the data and labels

        :param test_size: float, [0,1], optional:the ratio to split the test data. default is 0.2
        :param random_seed: float, optional:state a random seed for reproducible results, default is None
        :param shuffle: bool, optional:whether to shuffle the data, default is None
        :return: None
        '''
        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(load_iris().data, load_iris().target,
                             test_size=test_size, random_state=random_seed, shuffle=shuffle)

    def dist_2p(self, i, j):
        '''
        Calculates the distance between two points in n dimensions

        :param i: array: First vector
        :param j: array: Second vector
        :return: float: The resulting distance as float
        '''
        self.dist_result = 0
        if i.shape == j.shape:
            for dim in range(i.shape[0]):
                self.dist_result += np.square(i[dim] - j[dim])
            self.dist_result = np.sqrt(self.dist_result)
            return self.dist_result
        else:
            print("n dim of two points does not match!, check you're data")

    def predict_knn(self, test_elem, k, return_class=False):
        '''
        performs prediction of sample using K nearest neighbours

        :param test_elem: array: Object to perform prediction on
        :param k: int: number of K
        :param return_class: bool, optional: param for retrieving class for result or not,
            default is False
        :return: string of class if return_class if True and int of prediction if False
        '''
        knn_dist = np.zeros((len(self.train_data), 1))
        for i, elem in enumerate(self.train_data):
            knn_dist[i] = self.dist_2p(elem, test_elem)
        knn_array = np.column_stack((self.train_data, knn_dist, self.train_label))
        knn_array = knn_array[knn_array[:, -2].argsort()]
        if return_class:
            return self.categories[int(np.unique(knn_array[:k, -1], return_counts=True)[0][0])]
        else:
            return int(np.unique(knn_array[:k, -1], return_counts=True)[0][0])

    def test_accuracy(self, k):
        '''
        Performs K nearest neighbour analysis on all the test dataset on the train dataset.
        Returns accuracy and result

        :param k: int: number of K nearest neighbours to test
        :return: float , list: first element is the accuracy result, second element is a list of the label predictions
        '''
        self.test_results = []
        accuracy = 0
        for i, obj in enumerate(self.test_data):
            self.test_results.append(self.predict_knn(obj, k))
            accuracy += self.test_results[i] == self.test_label[i]
        return accuracy / len(self.test_results) * 100, self.test_results


dataset = Knn_ds(load_iris().data, load_iris().target, load_iris().target_names)
dataset.split(0.2, 42, True)
dataset.dist_2p(dataset.train_data[0], dataset.train_data[1])
print(dataset.predict_knn(dataset.train_data[0], 10, True))
print(dataset.test_accuracy(10))
