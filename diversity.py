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

total_results_forsplits = []
total_results_100 = []
for files in glob.glob('preprocessed_files/*'):
    print(files)
    for items in glob.glob("{}/*.*".format(files)):
        print(items)

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
            # print(my_model)
            # print(len(my_model))

            data = np.load(items, allow_pickle=True)
            # print(data)
            input_list = data.tolist()
            # x_train = input_list['FrameStack'][0]
            x_test = input_list['FrameStack'][1]
            # y_train = input_list['FrameStack'][2]
            y_test = input_list['FrameStack'][3]
            # print(x_train)
            # print(len(y_test))

            for i in range(1,100):
                y_pred2 = my_model[i].predict(x_test)
                y_pred1 = my_model[0].predict(x_test)

                results = deslib.util.diversity.Q_statistic(y_test, y_pred1, y_pred2)
                # results = deslib.util.diversity.double_fault(y_test, y_pred1, y_pred2)
                # results = deslib.util.diversity.ratio_errors(y_test, y_pred1, y_pred2)
                # results = deslib.util.diversity.disagreement_measure(y_test, y_pred1, y_pred2)
                # print(results)
                total_results_100.append(results)
        results_for_each_split= mean(total_results_100)

        # print(results_for_each_split)
        total_results_forsplits.append(results_for_each_split)
    # print(total_results_forsplits)
    results_for_each_dataset = mean(total_results_forsplits)
    print(results_for_each_dataset)






