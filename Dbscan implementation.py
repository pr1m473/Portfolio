import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

class Dbscan:
    def __init__(self,x,epsilon,minpts):
        self.x = x
        self.epsilon = epsilon
        self.minpts = minpts
        self.first_point = np.random.sample()