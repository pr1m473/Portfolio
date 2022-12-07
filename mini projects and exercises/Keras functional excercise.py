# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:35:41 2022

@author: BEN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
from PIL import Image
from tensorflow import keras
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
#import autokeras as ak
import seaborn as sns
from sklearn.utils import resample

# # -------------------------------------------------
# # first case with 7 categorical labeling no balancing,augmentation or layer optimization
#
# #preprocessing the metafile:
# meta = pd.read_csv(r'G:\My Drive\Primerose 18\Keras\HAM10000_metadata.csv')
# meta = meta.drop_duplicates(subset='lesion_id', keep='first').reset_index(drop=True)
# meta['age'] = meta['age'].fillna(meta['age'].mean())
# meta = meta[meta['sex'] != 'unknown']
# meta = meta[meta['localization'] != 'unknown']
# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#               for x in glob(os.path.join(r"G:\My Drive\Primerose 18\Keras", '*', '*.jpg'))}
# meta['path'] = meta['image_id'].map(image_path.get)
# meta['image'] = meta['path']. \
#     map(lambda x: np.asarray(Image.open(x).resize((100, 100))))
meta = pd.read_csv(r'G:\My Drive\Primerose 18\Keras\meta.csv')
shuffle_meta = meta.sample(frac=1)

x_data = np.hstack([pd.get_dummies(shuffle_meta['dx_type']).to_numpy(),
                   shuffle_meta['age'].values.reshape(len(shuffle_meta['age']),1),
                    pd.get_dummies(shuffle_meta['sex']).to_numpy()[:,0].reshape(len(meta),1),
                   pd.get_dummies(shuffle_meta['localization']).to_numpy()])

x_pic = shuffle_meta['image'].to_numpy()
def fix_x(x)
y_init = pd.get_dummies(shuffle_meta['dx']).to_numpy()


x_pic_train, x_pic_val, x_data_train, x_data_val, y_train, y_val = \
    train_test_split(x_pic, x_data, y_init, train_size=0.9,random_state = 42)

x_pic_train, x_pic_test, x_data_train, x_data_test, y_train, y_test = \
    train_test_split(x_pic_train, x_data_train, y_train, train_size=0.9,random_state = 42)


from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import visualkeras
from PIL import ImageFont

# Define the Picture (CNN) Stream

input_pic = layers.Input(shape=(100, 100, 3))
x1        = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))(input_pic)
x1        = layers.MaxPooling2D((2, 2))(x1)
x1        = layers.Conv2D(64, (3,3), activation='relu')(x1)
x1        = layers.MaxPooling2D((2, 2))(x1)
x1        = layers.Conv2D(64, (3,3), activation='relu')(x1)
x1        = layers.Flatten()(x1)
x1        = layers.Dense(7, activation='relu')(x1)
x1         = Model(inputs=input_pic, outputs=x1)

# Define the Stats (Feed-Forward) Stream

input_data = layers.Input(shape=(,20))
x2 = layers.Dense(64, activation="relu")(input_data)
x2 = layers.Dense(7, activation="relu")(x2)
x2 = Model(inputs=input_data, outputs=x2)

# Concatenate the two streams together
combined = layers.concatenate([x1.output, x2.output])

# Define joined Feed-Forward Layer
z = layers.Dense(4, activation="relu")(combined)

# Define output node of 7 neuron (regression task)
z = layers.Dense(7, activation="softmax")(z)

# Define the final model
model = Model(inputs=[x1.input, x2.input], outputs=z)
#image = visualkeras.layered_view(model, legend=True)


# Compile the model with Adam optimizer and mean-squared-error loss function

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mean_absolute_error'])

# Create a model saving callback and train for 10 epochs (connect to GPU runtime!!)
from tensorflow.keras.callbacks import ModelCheckpoint
cp = ModelCheckpoint('model/', save_best_only=True)
model.fit(x=[x_pic_train, x_data_train], y=y_train, validation_data=([x_pic_val, x_data_val], y_val), epochs=10, callbacks=[cp])

# #building the meta input array:
# input_meta_cat = np.ones((len(meta),4))
# le = LabelEncoder()
# le.fit(meta['dx_type'])
# input_meta_cat[:,0] = le.transform(meta['dx_type'])
# le.fit(meta['sex']) #need to remove unknown!
# input_meta_cat[:,1] = le.transform(meta['sex'])
# le.fit(meta['localization'])
# input_meta_cat[:,2] = le.transform(meta['localization'])

# input_meta_dummy = np.hstack((to_categorical(input_meta_cat[:,0]),
#                               input_meta_cat[:,1].reshape(len(input_meta_cat[:,1]),1),
#                               to_categorical(input_meta_cat[:,2]),
#                               meta['age'].values.reshape(len(meta['age']),1)))




#
# meta_array = pd.array([dx_type_cat.to_categorical()])
#
#  x_train,x_test,y_train,y_test = train_test_split(meta,
#                                                    meta['dx'],
#                                                    test_size=0.1
#                                                    random_state=42,
#                                                    stratify=meta['dx'])

# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#               for x in glob(os.path.join(r"G:\My Drive\Primerose 18\Keras", '*', '*.jpg'))}
#
# meta['path'] = meta['image_id'].map(image_path.get)
# meta_unique['image'] = meta_unique['path']. \
#     map(lambda x: np.asarray(Image.open(x).resize((32, 32))))
#
# df_0 = meta_unique[meta_unique['label'] == 0]
# df_1 = meta_unique[meta_unique['label'] == 1]
#
# df_0_balanced = resample(df_0, replace=True, n_samples=mel_count, random_state=42)
# df_1_balanced = resample(df_1, replace=True, n_samples=mel_count, random_state=42)
# meta_balanced = pd.concat([df_0_balanced, df_1_balanced])
#
# X = np.asarray(meta_balanced['image'].tolist())
# X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods
# Y = meta_balanced['label']  # Assign label values to Y
#
# # Testing for best predicted model
#
# x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y, test_size=0.90, random_state=42)
#
# # Further split data into smaller size to get a small test dataset.
# # x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.05, random_state=42)
#
# # Define classifier for autokeras. Here we check 25 different models, each model 25 epochs
# clf = ak.ImageClassifier(max_trials=25)  # MaxTrials - max. number of keras models to try
# clf.fit(x_train_auto, y_train_auto, epochs=10)
#
# # Evaluate the classifier on test data
# _, acc = clf.evaluate(x_test_auto, y_test_auto)
# print("Accuracy = ", (acc * 100.0), "%")
#
# # get the final best performing model
# model = clf.export_model()
# print(model.summary())
#
# # Save the model
# model.save('cifar_model.h5')
#
# score = model.evaluate(x_test_auto, y_test_auto)
# print('Test accuracy:', score[1])
#
# # ----------------------------------------------------------------
# # Running with all dx catagories (7 output nuerons in total)
# le = LabelEncoder()
# le.fit(meta_unique['dx'])
# print(list(le.classes_))
#
# meta_unique['label'] = le.transform(meta_unique["dx"])
# print(meta_unique.sample(10))
#
# df_0 = skin_df[skin_df['label'] == 0]
# df_1 = skin_df[skin_df['label'] == 1]
# df_2 = skin_df[skin_df['label'] == 2]
# df_3 = skin_df[skin_df['label'] == 3]
# df_4 = skin_df[skin_df['label'] == 4]
# df_5 = skin_df[skin_df['label'] == 5]
# df_6 = skin_df[skin_df['label'] == 6]
#
# n_samples = 500
# df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
# df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
# df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
# df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
# df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
# df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
# df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)
#
# skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
#                               df_2_balanced, df_3_balanced,
#                               df_4_balanced, df_5_balanced, df_6_balanced])
#
# X = np.asarray(meta_unique['image'].tolist())
# X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
# Y = meta_unique['label']  # Assign label values to Y
# Y_cat = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem
#
