import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



class Node:

    def __init__(self, depth, data=None):
        self.data = data
        self.feature = None
        self.value = None
        self.groups = None
        self.score = None
        self.left = None
        self.right = None
        self.depth = depth
        self.leaf_label = None
        self.groups, self.feature, self.value, self.score = randf_get_split(data)

    def build_tree(self, max_depth,min_size=3):
        self.max_depth = max_depth
        if self.depth >= self.max_depth or self.groups[0].shape[0] < min_size or self.groups[1].shape[0] < min_size:
            unique_labels = np.unique(self.data[:, -1], return_counts=True)
            self.leaf_label = unique_labels[0][np.argmax(unique_labels[1])]
            return

        else:
            self.left = Node(self.depth + 1, self.groups[0])
            self.left.build_tree(self.max_depth)
            self.right = Node(self.depth + 1, self.groups[1])
            self.right.build_tree(self.max_depth)

    def _predictor(self, test_data):
        if self.leaf_label is not None:
            return self.leaf_label
        if test_data[self.feature] < self.value:
            return self.left._predictor(test_data)
        else:
            return self.right._predictor(test_data)



    def print_tree(self):
        print('\t' * self.depth, self.depth)
        if self.right:
            print('R')
            self.right.print_tree()
        if self.left:
            print('L')
            self.left.print_tree()


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
            score += p * p
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


def randf_get_split(train_data):
    train_data = np.array(train_data)
    num_features = len(train_data[0])-1
    features = np.random.choice(range(num_features),size=int(np.sqrt(num_features)),replace=False)
    class_values = np.unique(train_data[:,-1])
    b_feature, b_value, b_score, b_groups = 0, 0, 999, None
    for feature in features:
        train_data = train_data[train_data[:, feature].argsort()]
        for row in train_data:
            groups = split(train_data, feature, row[feature])
            gini = calc_gini(groups, class_values)
            if gini < b_score:
                b_feature, b_value, b_score, b_groups = feature, row[feature], gini, groups
    return b_groups, b_feature, b_value, b_score

def create_subsample(dataset,alpha=1):
    if alpha>1:
        print('Alpha parameter must be <= 1')
        return
    sample_size = int(alpha*len(dataset))
    subsample_index = np.random.randint(0,len(dataset),size=sample_size)
    dataset_subsample = dataset[subsample_index]
    return dataset_subsample


def random_forest(train_data,m,alpha,max_depth=5,min_size=3):
    trees = []
    for i in range(m):
        sub_data = create_subsample(train_data,alpha)
        tree = Node(0,sub_data)
        tree.build_tree(max_depth=max_depth,min_size=min_size)
        trees.append(tree)
    return trees

def randf_predictor(data,trees):
    predictions = []
    for tree in trees:
        predictions.append(tree._predictor(data))
    best_predict = np.unique(predictions,return_counts=True)
    best_predict = best_predict[0][np.argmax(best_predict[1])]
    return best_predict

def randf_predict(test_data,trees,train_data,compare=True):
    predictions = []
    for data in test_data:
        predictions.append(randf_predictor(data,trees))
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += predictions[i] == test_data[i, -1]
    accuracy = accuracy / len(test_data) * 100

    print(f'The resulting predictions are:\n{predictions}')
    print(f'The calculated accuracy is: {np.round(accuracy, 2)}%')

    if compare:
        sk_tree = RandomForestClassifier(n_estimators=n_trees,min_samples_split=min_samples,max_depth=max_depth)
        sk_tree.fit(train_data[:, :-1], train_data[:, -1])
        sk_score = sk_tree.score(test_data[:, :-1], test_data[:, -1])
        print(f'The sklearn score is: {sk_score}')
    return accuracy

def K_fold_validator(data,k=5):
    data_size = len(data)
    data_indices = list(range(data_size))
    np.random.shuffle(data_indices)
    k_step = int(data_size/k)
    k_list = []
    for i in range(k):
        if i != k-1:
            k_list.append(data_indices[i * k_step:(i + 1) * k_step])
        else:
            k_list.append(data_indices[i * k_step:])
    scores = []
    for test_data in k_list:
        mask = np.ones(data_size, dtype=bool)
        mask[test_data] = False
        rand_forest = random_forest(data[mask],n_trees,0.5,max_depth,min_samples)
        scores.append(randf_predict(data[test_data], rand_forest, compare=True,train_data=data[mask]))
    print(f'The k-fold scores for k = {k} are:\n{scores}\nMean: {np.mean(scores)}\nStd: {np.std(scores)}')

if __name__ == '__main__':
    data_df = pd.read_csv('G:\My Drive\Primerose 18\Decision trees\wdbc.data')
    train_data, test_data = train_test_split(np.hstack((data_df.iloc[:, 2:],
                                                        data_df.iloc[:, 1].values.reshape(-1, 1))),
                                             test_size=0.2, shuffle=False)
    n_trees = 20
    min_samples = 2
    max_depth = 7

    # rand_forest = random_forest(train_data,n_trees,0.5,max_depth,min_samples)
    # randf_predict(test_data,rand_forest,True)
    data = np.hstack((data_df.iloc[:, 2:],data_df.iloc[:, 1].values.reshape(-1, 1)))
    K_fold_validator(data,k=5)
    print('end')









