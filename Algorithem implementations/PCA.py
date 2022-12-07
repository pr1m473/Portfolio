import os
import glob
import numpy as np
from imageio import imread_v2
from imageio import imwrite
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.pyplot import imshow

#Reading the Images
#-----------------------
image_paths = []
for root,dirs, files in os.walk(r"G:\My Drive\Primerose 18\Dimensionality reduction\PCA\faces941", topdown=True):
    for dir in dirs:
        image_list = glob.glob((os.path.join(root,dir,'*.jpg')))
        if len(image_list) != 0:
            image_paths.append(np.random.choice(image_list))
image_shape = imread_v2(image_paths[0]).shape
pic_pixels = image_shape[0]*image_shape[1]*image_shape[2]
image_array = np.zeros((len(image_paths),pic_pixels))
for i,path in enumerate(image_paths):
    image_array[i] = imread_v2(path).reshape((1,-1))

# #PCA with sklearn
# #---------------------------
#
# n_components = 100
# pca_object = PCA(n_components=n_components).fit(image_array)
# test_image = pca_object.transform(image_array[0].reshape(1,-1))
# test_image = pca_object.inverse_transform(test_image)
# imshow(test_image.reshape(image_shape).astype('int'))
#
# eigenfaces_array = pca_object.components_.reshape(100,image_shape[0],image_shape[1],image_shape[2])
# for i in range(len(eigenfaces_array)):
#     imwrite(r'G:\My Drive\Primerose 18\PCA\eigenfaces'+'/'+str(i)+'.jpg',eigenfaces_array[i])
#

#PART 2: PCA from scratch
class PCA:
    def __init__(self,n_components,X):
        self.n_components = n_components
        self.X = np.array(X)
        self.X_centered = self.X - self.X.mean(axis=0)

    def fit(self):
        self.cov_matrix = np.cov(self.X_centered) # using a smaller dimension trick
        self.eigen_val, self.eigen_vect = np.linalg.eig(self.cov_matrix)
        #Sort the eigen values and vectors
        sorted_index = np.argsort(self.eigen_val)[::-1]
        self.eigen_val = self.eigen_val[sorted_index]
        self.eigen_vect = self.eigen_vect[sorted_index,:]
        self.eigen_vect = self.eigen_vect[0:self.n_components,:]
        self.eigen_vect = self.eigen_vect @ self.X_centered
        # self.eigen_vect = minmax_scale(self.eigen_vect,axis=1)


    def transform(self,data):
        norm_data = data.reshape(1,-1) - self.X.mean(axis=0)
        return norm_data @ self.eigen_vect.T

    def reverse_transform(self,data):
        return data @ self.eigen_vect

my_pca = PCA(100,image_array)
my_pca.fit()
test1 = my_pca.transform(image_array[120])
test2 = my_pca.reverse_transform(test1)
test3 = minmax_scale(test2,axis=1)
imshow(test3.reshape(image_shape))
print()










