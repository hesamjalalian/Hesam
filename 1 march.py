import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import ShuffleSplit
import pickle
import cv2
import glob
from sklearn.model_selection import StratifiedShuffleSplit

file_list = glob.glob('datasets/*.*')
#print(file_list)

# my_list = []   # to store them
# path = "datasets/*.*"

for file in glob.glob("datasets/*.*"):
    #print(file)
    # a = cv2.imread(file)
    # my_list.append(a)

    #dataset = loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Banana.mat")
    #data = dataset['dataset']
    # print(my_list)

    input = loadmat(file)
    # print(file)

    data = input['dataset']
    # header = input["__header__"]
    # version = input["__version__"]
    # globals = input["__globals__"]
    # print(header)
    # print(version)
    # print(globals)
    # print(data)

    X = data[:, 0:-1]
    y = data[:, -1]
    # print(X.shape) # how to get the array shape
    #df = sns.load_dataset('tips')
    #new_data_label = df
    #data = pd.get_dummies(data[:, -1])

    #rs = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    #for train_index, test_index,  in rs.split(X):
    #    print("TRAIN:", train_index, "TEST:", test_index)

    #X_train = []
    #X_test = []
    #print(train_index)
    #print(y_train)

    #train_test_split(X, y, test_size=0.2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    split_part=0
    name = file.split(".")[0].split("\\")[1]
    # print(name)

    #how to creat directory
    # dir = "/{}_dir".format(name)
    # parent_dir = "C:Users/Hesam/git_repo/Hesam"
    # path = os.path.join(parent_dir,dir)
    # os.makedirs(path, exist_ok=True)
    # print(os.getcwd())

    for train_index, test_index in sss.split(X, y):
        split_part=split_part+1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train)

        #X_train = []
        #X_test = []

        label_encoder = preprocessing.LabelEncoder()
        y_train_ms = label_encoder.fit_transform(y_train)
        y_test_ms = label_encoder.transform(y_test)

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
        #print(y_test_ms)
        #print(y_train)

        X_train_processed = X_train_ms
        X_test_processed = X_test_ms
        y_train_processed = y_train_ms
        y_test_processed = y_test_ms

        processed = [X_train_processed, X_test_processed, y_train_processed, y_test_processed]
        #print(file)
        name = file.split(".")[0].split("\\")[1]
        # print(name)

        # How to savedata in npz format
        # np.savez('mat{}.npz'.format(name), *processed)

        # How to savedata in mat format
        FramStack = np.empty((len(processed),), dtype=object)
        for i in range(len(processed)):
            FramStack[i]=processed[i]

        savemat('{}_dir/processed_{}_{}.mat'.format(name,name,split_part),{"FrameStack":FramStack})

        # with open ('processed2222.pkl', 'wb') as processed_output:
        #     pickle.dump(processed, processed_output, pickle.HIGHEST_PROTOCOL)
        # print(X_train_processed)
        #print(y_test_processed)



    #processed = ['X_train_ms, X_test_ms, y_train, y_test']
    #with open ('processed.pkl', 'wb') as processedpickle:
    #    pickle.dump(processed, processedpickle)