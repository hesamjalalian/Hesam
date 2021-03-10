import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import ShuffleSplit

dataset = loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Wine.mat")
data=dataset['dataset']
#print(dataset)

X = data[:, 0:-1]
y = data[:, -1]

label_encoder = preprocessing.LabelEncoder()
data[:, -1] = label_encoder.fit_transform(data[:, -1])

# I tried here to split once and I am going to update it with 20 repetition
train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print(X_train)
#print(X_train)

# normalization
ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.transform(X_test)

#print(X_test_ms)

# handling the missing data and replace missing values with mean of all the other values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train_ms[:, 1:])
X_train_ms[:, 1:] = imputer.transform(X_train_ms[:, 1:])
X_test_ms[:, 1:] = imputer.transform(X_test_ms[:, 1:])
print(X_train_ms)
print(X_test_ms)

#label_encoder = preprocessing.LabelEncoder()
#data[:, -1] = label_encoder.fit_transform(data[:, -1])
#print(data[:, -1])
#print(y_test)
#print(y_train)

x_train_processed = X_train_ms
x_test_processed = X_test_ms
y_train_processed = y_train
y_test_processed = y_test
#print(y_train_processed)

######   Hesam save data ##################how to save multiple NumPy array variables in a single file########################################

