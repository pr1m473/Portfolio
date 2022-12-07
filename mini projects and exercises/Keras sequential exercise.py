# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:52:38 2022

@author: BEN
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt(r'G:\My Drive\Primerose 18\Keras\diabetes.csv', delimiter=',',skiprows=1)
X = dataset[:,0:8]
y = dataset[:,8]

#Defining the keras model
model = Sequential()
model.add(Dense(8, input_shape=(8,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X, y, epochs=150, batch_size=10)

#Evaluate the model
_,accuracy = model.evaluate(X, y) #The _ syntax will ignore loss as we're not intrested in that

print('Accuracy: %.2f' % (accuracy*100))

#Preictions using the model
predictions = (model.predict(X) > 0.5).astype(int)

for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
