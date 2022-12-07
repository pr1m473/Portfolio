import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import Random_Forest\Random_forest

class Node:

    def __init__(self, depth,data=None):
        self.data = data
        self.feature = None
        self.value = None
        self.groups = None
        self.score = None
        self.left = None
        self.right = None
        # self.children=[]
        self.depth = depth
        self.leaf_label = None
        self.groups, self.feature, self.value, self.score = get_split(data)


    def build_tree(self, max_depth):
        self.max_depth = max_depth
        if self.depth >= self.max_depth or self.groups[0].shape[0] == 0 or self.groups[1].shape[0] == 0:
            unique_labels = np.unique(self.data[:,-1],return_counts=True)
            self.leaf_label = unique_labels[0][np.argmax(unique_labels[1])]
            return

        else:
            self.left = Node(self.depth + 1,self.groups[0])
            self.left.build_tree(self.max_depth)
            self.right = Node(self.depth + 1,self.groups[1])
            self.right.build_tree(self.max_depth)

    def _predictor(self,test_data):
        if self.leaf_label is not None:
            return self.leaf_label
        if test_data[self.feature] < self.value:
            return self.left._predictor(test_data)
        else:
            return self.right._predictor(test_data)

    def predict(self,test_dataset,compare=False):
        predictions = []
        accuracy = 0
        for test_data in test_dataset:
            predictions.append(self._predictor(test_data))
        for i in range(len(predictions)):
            accuracy += predictions[i] == test_dataset[i,-1]
        accuracy = accuracy / len(test_dataset) * 100

        print(f'The resulting predictions are:\n{predictions}')
        print(f'The calculated accuracy is: {np.round(accuracy,2)}%')

        if compare:
            sk_tree = DecisionTreeClassifier(max_depth=self.max_depth)
            sk_tree.fit(train_data[:,:-1],train_data[:,-1])
            sk_score =  sk_tree.score(test_dataset[:,:-1],test_dataset[:,-1])
            print(f'The sklearn score is: {sk_score}')



    def print_tree(self):
        print('\t' * self.depth, self.depth)
        if self.right:
            print('R')
            self.right.print_tree()
        if self.left:
            print('L')
            self.left.print_tree()

# def train_test_split(test_size=0.2, random_seed=None, shuffle=False):
#     '''
#     performs train_test_split on the data and labels
#
#     :param test_size: float, [0,1], optional:the ratio to split the test data. default is 0.2
#     :param random_seed: float, optional:state a random seed for reproducible results, default is None
#     :param shuffle: bool, optional:whether to shuffle the data, default is None
#     :return: None
#     '''
#     self.train, self.test = train_test_split(self.dataset, test_size=test_size,
#                                              random_state=random_seed, shuffle=shuffle)

def calc_gini(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p**2
        gini += (1 - score) * (size / n_instances)
    return gini

def split(data, split_index, split_value):
    left = []
    right = []
    for row in data:
        if row[split_index] < split_value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)

def get_split(train_data):
    train_data = np.array(train_data)
    class_values = np.unique(train_data[:, -1])
    b_feature, b_value, b_score, b_groups = 0, 0, 999, None
    for feature in range(len(train_data[0]) - 1):
        train_data = train_data[train_data[:, feature].argsort()]

        for row in train_data:
            groups = split(train_data, feature, row[feature])
            gini = calc_gini(groups, class_values)
            # print('X%d < %.3f Gini=%.3f' % ((feature + 1), row[feature], gini))
            if gini < b_score:
                b_feature, b_value, b_score, b_groups = feature, row[feature], gini, groups
    return  b_groups, b_feature, b_value, b_score





# if __name__ == '__main__':
#     root = Node(0)
#     root.build_tree(3)
#     root.print_tree()

if __name__ == '__main__':
    data_df = pd.read_csv('G:\My Drive\Primerose 18\Decision trees\wdbc.data')
    train_data, test_data =  train_test_split(np.hstack((data_df.iloc[:,2:],
                                                         data_df.iloc[:,1].values.reshape(-1,1))),
                                              test_size=0.2,shuffle=False)
    temp_tree = Node(0,train_data)
    temp_tree.build_tree(5)
    temp_tree.predict(test_data,True)

    # dataset = np.array([[2.771244718, 1.784783929, 0],
    #            [1.728571309, 1.169761413, 0],
    #            [3.678319846, 2.81281357, 0],
    #            [3.961043357, 2.61995032, 0],
    #            [2.999208922, 2.209014212, 0],
    #            [7.497545867, 3.162953546, 1],
    #            [9.00220326, 3.339047188, 1],
    #            [7.444542326, 0.476683375, 1],
    #            [10.12493903, 3.234550982, 1],
    #            [6.642287351, 3.319983761, 1]])
    # test_tree = Decision_tree(dataset[:,:-1],dataset[:,-1],split=False)
    # print(tree_model.get_split())
    print('end')
