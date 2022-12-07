import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial import distance_matrix

class K_means:
    def __init__(self,x,k):
        self.x = x
        self.k = k
        self.means = self.x[np.random.randint(0, len(self.x) - 1, (k)), :]
        # self.means = np.random.uniform(np.min(x),np.max(x),(k,x.shape[1]))
        self.cluster_dict = {}
        self.thershold_flag = False
        for i in range(self.k):
            self.cluster_dict[i] = []

    def _get_clusters(self):
        dist_matrix  = distance_matrix(self.x,self.means,p=2)
        for i in range(len(self.x)):
            self.cluster_dict[np.argmin(dist_matrix[i])].append(i)

    def _get_means(self):
        new_means = self.means.copy()
        for i in range(self.k):
            new_means[i] = np.mean(self.x[self.cluster_dict[2]],axis=0)
        if max(distance_matrix(new_means,self.means,p=2)[0]) < 1:
            self.thershold_flag = True
        self.means = new_means

    def _clusters_plot(self):
        # ax.scatter(self.cluster_dict[:,:1], c=df['continent'].map(colors))

    def fit(self):
        while self.thershold_flag == False:
            self._get_clusters()
            self._get_means()




if __name__ == '__main__':
    iris_data = load_iris().data
    kmodel_iris = K_means(iris_data,3)
    kmodel_iris.fit()

    print()



