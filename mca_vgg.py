# -*- coding: utf-8 -*-
"""mca_vgg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11hzEOow9qR3l7BaTjZuywDH5QDDYxc5P
"""

!pip uninstall tensorflow
!pip install tensorflow==1.14.0 --ignore-installed

!unzip '/content/drive/My Drive/mca/MCA_Project/train_frames.zip' -d '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/'

# Commented out IPython magic to ensure Python compatibility.
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
# %matplotlib inline
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
from keras import regularizers
from keras.optimizers import Nadam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

train_image = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/frames_dev_final.pkl')
to_rows = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dia_frames_dev_dia_to_row.pkl', allow_pickle=True).item()
Y = np.zeros((train_image.shape[0], 1))

import pandas as pd

test_csv = pd.read_csv(r'/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dev_sent_emo.csv')
test_csv = np.array(test_csv)

#print(train_image.shape)

for nm, k in enumerate(to_rows.keys()):
  u = k.find('_')
  dia = k[:u]
  utt = k[u+1:]

  emo = 0

  for t in test_csv:
      if str(t[5]) == dia and str(t[6]) == utt:
        emo = t[3]
        break

  emo_nm = 0
  if emo == 'neutral' :
    emo_nm = 0
  elif emo == 'sadness':
    emo_nm = 1
  elif emo == 'anger':
    emo_nm = 2
  elif emo == 'joy':
    emo_nm = 3

  for t in to_rows[k]:
    Y[t] = emo_nm

print(Y.shape)

X = np.array(train_image)

outfile = open('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_dev.pkl', 'wb')
np.save(outfile, Y)

print(X)

base_model = VGG16(weights='imagenet', include_top=False)

print("Loaded model")
X = base_model.predict(X)

print(X.shape)

#outfile = open('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_dev.pkl', 'wb')
#np.save(outfile, X)

outfile = open('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_train.pkl', 'wb')
np.save(outfile, Y)

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y = encoder.fit_transform(Y)
print(Y.shape)

from keras.callbacks import ModelCheckpoint, EarlyStopping

X = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_train.pkl') 

mx = X.max()
X = X/mx

X = X.reshape(49684, 2*2*512)

print(X.shape, Y.shape)

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

from keras.optimizers import Nadam

optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)
early_stopping = EarlyStopping(monitor='accuracy', patience=10)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(X, Y, epochs=25, batch_size=100, callbacks=[early_stopping], class_weight = [1.0, 3.0, 3.0, 3.0])


#import pickle

#filename = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_1000.sav'
#pickle.dump(model, open(filename, 'wb'))

X_test = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/frames_test_final.pkl')

to_rows = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dia_frames_test_dia_to_row.pkl', allow_pickle=True).item()
Y_test = np.zeros((X_test.shape[0], 1))

import pandas as pd

test_csv = pd.read_csv(r'/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/test_sent_emo.csv')
test_csv = np.array(test_csv)

print(X_test.shape)

for nm, k in enumerate(to_rows.keys()):
  u = k.find('_')
  dia = k[:u]
  utt = k[u+1:]

  emo = 0

  for t in test_csv:
      if str(t[5]) == dia and str(t[6]) == utt:
        emo = t[3]
        break

  emo_nm = 0
  if emo == 'neutral' :
    emo_nm = 0
  elif emo == 'sadness':
    emo_nm = 1
  elif emo == 'anger':
    emo_nm = 2
  elif emo == 'joy':
    emo_nm = 3

  for t in to_rows[k]:
    Y_test[t] = emo_nm

print(Y_test.shape)

X_test = np.array(X_test)

#outfile = open('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_test.pkl', 'wb')
#p.save(outfile, Y_test)

base_model = VGG16(weights='imagenet', include_top=False)

X_test = base_model.predict(X_test)
#outfile = open('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_test.pkl', 'wb')
#np.save(outfile, X_test)
print(X_test.shape)

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y_test = encoder.fit_transform(Y_test)
print(Y_test.shape)

X_test = X_test.reshape(8770, 2048)
Y_pred = model.predict(X_test)

Y_p = np.zeros(Y_pred.shape)

for nt, yt in enumerate(Y_pred):
  Y_p[nt, np.argmax(yt)] = 1

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

true_label = []
predicted_label = []

for yt in Y_test:
  true_label.append(np.argmax(yt))

for yp in Y_p:
  predicted_label.append(np.argmax(yp))
  
print("Confusion Matrix :")
print(confusion_matrix(true_label, predicted_label))
print("Classification Report :")
print(classification_report(true_label, predicted_label, digits=4))
print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

X_train = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_train.pkl', allow_pickle=True)
X_dev = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_dev.pkl', allow_pickle=True)
X_test = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_x_test.pkl', allow_pickle=True)

Y_train = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_train.pkl', allow_pickle=True)
Y_dev = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_dev.pkl', allow_pickle=True)
Y_test = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_y_test.pkl', allow_pickle=True)

to_row_train = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dia_frames_train_dia_to_row.pkl', allow_pickle=True).item()
to_row_dev = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dia_frames_dev_dia_to_row.pkl', allow_pickle=True).item()
to_row_test = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/dia_frames_test_dia_to_row.pkl', allow_pickle=True).item()

#print(X_train.shape, Y_train.shape)
#print(X_dev.shape, Y_dev.shape)
#print(X_test.shape, Y_test.shape)

X_train_final = []
X_dev_final = []
Y_train_final = []
Y_dev_final = []

count_emotions = np.zeros((4, 1))

for k in to_row_train.keys():

  emo = int(Y_train[to_row_train[k][0]])
  
  if count_emotions[emo] < 1500:
    count_emotions[emo] += 1

    for t in to_row_train[k]:
      X_train_final.append(X_train[t])
      Y_train_final.append(Y_train[t])

X_train_final = np.array(X_train_final)
Y_train_final = np.array(Y_train_final)

count_emotions = np.zeros((4, 1))

for k in to_row_dev.keys():

  emo = int(Y_dev[to_row_dev[k][0]])
  
  if count_emotions[emo] < 200:
    count_emotions[emo] += 1

    for t in to_row_dev[k]:
      X_dev_final.append(X_dev[t])
      Y_dev_final.append(Y_dev[t])

X_dev_final = np.array(X_dev_final)
Y_dev_final = np.array(Y_dev_final)


#model = Sequential()
#model.add(Dense(1024, activation='relu', input_shape=(2048,), kernel_regularizer=regularizers.l2(0.00001),activity_regularizer=regularizers.l1(0.00001)))
#model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001),activity_regularizer=regularizers.l1(0.00001)))
#model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001),activity_regularizer=regularizers.l1(0.00001)))
#model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001),activity_regularizer=regularizers.l1(0.00001)))
#model.add(Dense(4, activation='softmax'))

#model.summary()

model = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_balanced.sav', allow_pickle= True)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

#model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

X_train_final = X_train_final.reshape(X_train_final.shape[0], 2048)
X_test = X_test.reshape(X_test.shape[0], 2048)
X_dev_final = X_dev_final.reshape(X_dev_final.shape[0], 2048)


encoder = LabelBinarizer()
Y_test = encoder.fit_transform(Y_test)

encoder = LabelBinarizer()
Y_train_final = encoder.fit_transform(Y_train_final)

encoder = LabelBinarizer()
Y_dev_final = encoder.fit_transform(Y_dev_final)


#model.fit(X_train_final, Y_train_final, epochs=25, batch_size=100, callbacks=[early_stopping], validation_data = [X_dev_final, Y_dev_final]) #class_weight = [1.0, 3.0, 3.0, 3.0])

Y_pred = model.predict(X_test)

Y_p = np.zeros(Y_pred.shape)

for nt, yt in enumerate(Y_pred):
  Y_p[nt, np.argmax(yt)] = 1

true_label = []
predicted_label = []

for yt in Y_test:
  true_label.append(np.argmax(yt))

for yp in Y_p:
  predicted_label.append(np.argmax(yp))
  
print("Confusion Matrix :")
print(confusion_matrix(true_label, predicted_label))
print("Classification Report :")
print(classification_report(true_label, predicted_label, digits=4))
print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

'''
fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/x_train_final_vgg.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, X_train_final)

fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/x_dev_final_vgg.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, X_dev_final)

fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/y_train_final_vgg.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, Y_train_final)

fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/y_dev_final_vgg.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, Y_dev_final)
'''

fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/y_test_vgg_final.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, Y_test)

fn_dict = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/x_test_vgg_final.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, X_test)

filename = '/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_balanced.sav'
pickle.dump(model, open(filename, 'wb'))

model = np.load('/content/drive/My Drive/mca/MCA_Project/MELD_Dataset/vgg_balanced.sav')