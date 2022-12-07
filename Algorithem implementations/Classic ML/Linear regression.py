
# Linear regression exercise

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x_size = 100
gauss_std = 1

# Get k random fibonacci number from size 'high' series
def get_rand_fib(high,k):
    num_arr = [int((((1 + np.sqrt(5)) / 2) ** (i+1)) / np.sqrt(5) + 0.5) for i in range(high)]
    return random.sample(num_arr,k)

def add_gauss_noise(array):
    array += np.random.normal(scale=gauss_std,size=(len(array),1))
    return array

def lin_regr(x_points,y_points):
    if x_points.ndim==1 or y_points.ndim==1:
        x_points = x_points.reshape[-1, 1]
        y_points = y_points.reshape[-1, 1]
    assert x_points.shape[0] == y_points.shape[0], "Dimensions of x and y don't match"
    assert not np.isclose(np.linalg.det(x_points.T@x_points),0), "matrix is not invertible!"
    return np.linalg.inv(x_points.T.dot(x_points)).dot(x_points.T).dot(y_points)

def build_plot(x_data,y_data,x_homog,y_pred,title):
    plt.figure()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_data[:, 0], y_data)
    plt.plot(x_homog, y_pred)
    plt.show()



int_array =  list(np.random.randint(1,10,size=x_size))
x = np.reshape(np.random.normal(size=x_size),(-1,1))
vect1 = np.random.randint(20,size=5)*3

y_line_1 = add_gauss_noise(x*random.sample(int_array,1))
hfunc1 = lin_regr(x,y_line_1)

x2 = np.column_stack((x,np.ones_like(x)))
y_line_2 = add_gauss_noise(x*random.sample(int_array,1) + random.sample(int_array,1))
hfunc2 = lin_regr(x2,y_line_2)

x3 = np.column_stack((x,x**2,np.ones_like(x)))
y_line_3 = add_gauss_noise(random.sample(int_array,1)*(x**2) + x*random.sample(int_array,1) + random.sample(int_array,1))
hfunc3= lin_regr(x3,y_line_3)

x4 = np.array([0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,
               0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,
               0.26468939,0.54419358,0.89099501,0.9591165,0.9496439,0.82249202,
               0.99367066,0.50628823]).reshape((-1,1))
y4 = np.array([4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,
      20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,
      12.46321171,47.70487443,65.70793999,62.7767844,35.22558438,
      77.8453303,11.08106882]).reshape((-1,1))
y4_ln = np.log(add_gauss_noise(y4))

x4 = np.column_stack((x4,x4**2,np.ones_like(x4)))
hfunc4 = lin_regr(x4,y4_ln)
a_hfunc4 = np.exp(hfunc4[2])



x_homog = np.linspace(-1,1).reshape(-1,1)
build_plot(x,y_line_1,x_homog,np.dot(x_homog,hfunc1),"Simple linear")


x_homog2 = np.column_stack((x_homog,np.ones_like(x_homog)))
build_plot(x2,y_line_2,x_homog2[:,0],np.dot(x_homog2,hfunc2),"Simple linear with intercept")


x_homog3 = np.column_stack((x_homog,x_homog**2,np.ones_like(x_homog)))
build_plot(x3,y_line_3,x_homog3[:,0],np.dot(x_homog3,hfunc3),"Polinom 2nd order ")

build_plot(x4,y4,x_homog,a_hfunc4*np.exp(hfunc4[0]*x_homog+hfunc4[1]*x_homog**2),"Exponent")








