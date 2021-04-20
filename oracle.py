import glob
import deslib
from deslib.static import Oracle
from sklearn.utils.validation import check_X_y, check_array
import self
from numpy import mean
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import pickle
from sklearn.linear_model import Perceptron
from tabulate import tabulate
from dataclasses import make_dataclass
from sklearn.ensemble import RandomForestClassifier
from deslib.static.base import BaseStaticEnsemble

dict_final_results_oracle = {}

for files in glob.glob('preprocessed_files/*'):
    print(files)
    for items in glob.glob("{}/*.*".format(files)):
        # print(items)
        oracleaccuracies = []
        accuracy = []
        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        # print(item_name)
        model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        # print(model_load)
        with open(model_load, 'rb') as file:
            my_model = pickle.load(file)

            data = np.load(items, allow_pickle=True)
            # print(data)
            input_list = data.tolist()  # How to get x_train
            x_train = input_list['FrameStack'][0]
            x_test = input_list['FrameStack'][1]
            y_train = input_list['FrameStack'][2]
            y_test = input_list['FrameStack'][3]
            # print(x_train)

            # print('Shape of X_train=>', x_train.shape)
            # print('Shape of X_test=>', x_test.shape)
            # print('Shape of Y_train=>', y_train.shape)
            # print('Shape of Y_test=>', y_test.shape)

            oracleeeee = deslib.static.oracle.Oracle(pool_classifiers=my_model, random_state=None,
                                                     n_jobs=-1)

            oracleeeee.fit(x_train, y_train)
            y_pred = oracleeeee.predict(x_test, y_test)
            accuracyyy = accuracy_score(y_test, y_pred)
            accuracy.append(accuracyyy)

    dict_final_results_oracle["{}".format(files)] = mean(accuracy)
print(dict_final_results_oracle)










