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
import numpy as np
import pickle
import cv2
import glob

file_list = glob.glob('datasets/*.*')
#print(file_list)

my_list=[]   # to store them

path = "datasets/*.*"
for file in glob.glob(path):
    #print(file)
    a = cv2.imread(file)
    my_list.append(a)

#dataset = loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Wine.mat")
#data=dataset['dataset']

dataset = loadmat(file)
data=dataset['dataset']
print(dataset)

X = data[:, 0:-1]
y = data[:, -1]

label_encoder = preprocessing.LabelEncoder()
data[:, -1] = label_encoder.fit_transform(data[:, -1])


#rs = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
#for train_index, test_index in rs.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)

#print(X_train)
#print(X_train)

train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print(X_train)
#print(X_test)
# normalization
ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.transform(X_test)
#print(X_train_ms)
#print(X_test_ms)

# handling the missing data and replace missing values with mean of all the other values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train_ms[:, 1:])
X_train_ms[:, 1:] = imputer.transform(X_train_ms[:, 1:])
X_test_ms[:, 1:] = imputer.transform(X_test_ms[:, 1:])
#print(X_train_ms)
#print(X_test_ms)
#print(y_test)
#print(y_train)

X_tran_processed = X_train_ms
X_test_processed = X_test_ms
y_train_processed = y_train
y_test_processed = y_test

processed = ['X_tran_processed, X_test_processed, y_train_processed, y_test_processed']
with open ('processed.pkl', 'wb') as processedpickle:
    pickle.dump(processed, processedpickle)

print(X_tran_processed)
