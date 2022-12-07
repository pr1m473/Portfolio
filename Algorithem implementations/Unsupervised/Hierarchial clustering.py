import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean


class H_clustering:
    def __init__(self, X, max_cluster, method='full'):
        self.method = {'full': self._full_linkage,
                       'single': self._single_linkage,
                       'wpgma': self._wpgma,
                       'upgma': self._upgma,
                       'ward': self._ward}
        self.method = self.method[method]
        self.max_cluster = max_cluster
        self.dist_matrix = distance_matrix(X, X, p=2)
        self.cluster_list = [list([i]) for i in range(len(X))]
        self.linkage = []

    def _full_linkage(self):
        cluster_dist_array = np.zeros((len(self.dist_matrix), 1))
        for i in range(len(self.dist_matrix)):
            cluster_dist_array[i] = max(self.dist_matrix[self.next_cluster[0], i],
                                        self.dist_matrix[self.next_cluster[1], i])
        cluster_dist_array = np.delete(cluster_dist_array, self.next_cluster[1])
        cluster_dist_array[self.next_cluster[0]] = 0
        return cluster_dist_array

    def _single_linkage(self):
        cluster_dist_array = np.zeros((len(self.dist_matrix), 1))
        for i in range(len(self.dist_matrix)):
            cluster_dist_array[i] = min(self.dist_matrix[self.next_cluster[0], i],
                                        self.dist_matrix[self.next_cluster[1], i])
        cluster_dist_array = np.delete(cluster_dist_array, self.next_cluster[1])
        cluster_dist_array[self.next_cluster[0]] = 0
        return cluster_dist_array

    def _wpgma(self):
        cluster_dist_array = np.zeros((len(self.dist_matrix), 1))
        for i in range(len(self.dist_matrix)):
            cluster_dist_array[i] = (self.dist_matrix[self.next_cluster[0], i] + \
                                     self.dist_matrix[self.next_cluster[1], i])/2
        cluster_dist_array = np.delete(cluster_dist_array, self.next_cluster[1])
        cluster_dist_array[self.next_cluster[0]] = 0
        return cluster_dist_array

    def _upgma(self):
        cluster_dist_array = np.zeros((len(self.dist_matrix), 1))
        cluster_sizes = [len(self.cluster_list[self.next_cluster[0]]),len(self.cluster_list[self.next_cluster[1]])]
        for i in range(len(self.dist_matrix)):
            cluster_dist_array[i] = (self.dist_matrix[self.next_cluster[0], i]*cluster_sizes[0] + \
                                     self.dist_matrix[self.next_cluster[1], i]*cluster_sizes[1]) / sum(cluster_sizes)
        cluster_dist_array = np.delete(cluster_dist_array, self.next_cluster[1])
        cluster_dist_array[self.next_cluster[0]] = 0
        return cluster_dist_array

    def _ward(self):
        # #We only now define distance as squared distance since first clustering will be the same
        # self.dist_matrix = self.dist_matrix ** 2
        # cluster_dist_array = np.zeros((len(self.dist_matrix), 1))
        # for i in range(len(self.dist_matrix)):
        #     cluster_dist_array[i] = (self.dist_matrix[self.next_cluster[0], i]*cluster_sizes[0] + \
        #                              self.dist_matrix[self.next_cluster[1], i]*cluster_sizes[1]) / sum(cluster_sizes)

    def _update_matrix(self):
        self.next_cluster = np.argwhere(self.dist_matrix == min(self.dist_matrix[self.dist_matrix > 0]))[0]
        cluster_dist_array = self.method()
        self.linkage.append(self.dist_matrix[self.next_cluster[0], self.next_cluster[1]] / 2)
        self.dist_matrix = np.delete(self.dist_matrix, self.next_cluster[1], axis=1)
        self.dist_matrix = np.delete(self.dist_matrix, self.next_cluster[1], axis=0)
        self.dist_matrix[self.next_cluster[0], :] = cluster_dist_array
        self.dist_matrix[:, self.next_cluster[0]] = cluster_dist_array

    def agglomerate(self):
        while len(self.cluster_list) > self.max_cluster:
            self._update_matrix()
            self.cluster_list[self.next_cluster[0]] += self.cluster_list[self.next_cluster[1]]
            self.cluster_list.pop(self.next_cluster[1])


if __name__ == '__main__':
    iris_data = load_iris().data
    iris_labels = load_iris().target
    iris_clustring = H_clustering(iris_data, 3,method='upgma')
    iris_clustring.agglomerate()
    print()
