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
import sys

# def oracle(x_train, y_train, model_name, pool_of_classifier):
#     oracleeeee = deslib.static.oracle.Oracle(pool_classifiers=pool_of_classifier, random_state=None, n_jobs=-1)
#     oracleeeee.fit(x_train, y_train)
#     y_pred = oracleeeee.predict(x_test, y_test)
#     # accuracy = accuracy_score(y_test, y_pred)
#     return y_pred
#
#
# def bagging_with_decision_tree(x_train, y_train, model_name):
#     dt = DecisionTreeClassifier(random_state=42)
#     Bagging_classifiers_DecisionTree_AsBaseEstimator = BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1, random_state=42)
#     Bagging_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
#     y_pred = Bagging_classifiers_DecisionTree_AsBaseEstimator.predict(x_test) # Predict test set labels
#     return y_pred
#
# def bagging_with_perceptron(x_train, y_train, model_name):
#     perc = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
#     Bagging_classifiers_Perceptron_AsBaseEstimator = BaggingClassifier(base_estimator=perc, n_estimators=100, n_jobs=-1, random_state=42)  # Instantiate a BaggingClassifier 'Bagging_classifiers_Perceptron_AsBaseEstimator'
#     split_num = model_name.split('_')[4].split('.')[0]
#     portion = model_name.split('\\')[1]
#     model = Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)
#     pkl_filename = "model_files/{}/split_{}/bagging_with_perceptron.pkl".format(portion, split_num)
#     with open(pkl_filename, 'wb') as file:
#         pickle.dump(model, file)
#     Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)  # Fit 'bc' to the training set
#     oracle_trained_model = Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)
#     # oracle_accuracy = oracle(x_train, y_train, model_name, oracle_trained_model)
#     y_pred = Bagging_classifiers_Perceptron_AsBaseEstimator.predict(x_test)  # Predict test set labels
#     # accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
#     # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
#     return y_pred
#

#
# def boosting_with_decision_tree(x_train, y_train, model_name):
#     bdt = DecisionTreeClassifier(random_state=42, max_depth=2)
#     boosting_classifiers_DecisionTree_AsBaseEstimator = AdaBoostClassifier(base_estimator=bdt, n_estimators=100, random_state=0, algorithm='SAMME')  # Instantiate a BoostingClassifier 'Boosting_classifiers_Perceptron_AsBaseEstimator'
#
#     split_num = model_name.split('_')[4].split('.')[0]
#     portion = model_name.split('\\')[1]
#     model = boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
#     pkl_filename = "model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(portion, split_num)
#     with open(pkl_filename, 'wb') as file:
#         pickle.dump(model, file)
#
#
#
#     # oracle_trained_model = boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
#     # oracle_accuracy = oracle(x_train, y_train, model_name, oracle_trained_model)
#
#     y_pred = boosting_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
#     # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
#     return accuracy
#
# def boosting_with_perceptron(x_train, y_train, model_name):
#     bperc = Perceptron(tol=1e-3, random_state=42)
#     boosting_classifiers_perceptron_AsBaseEstimator = AdaBoostClassifier(base_estimator=bperc, n_estimators=100, random_state=0, algorithm='SAMME')
#
#     split_num = model_name.split('_')[4].split('.')[0]
#     portion = model_name.split('\\')[1]
#     model = boosting_classifiers_perceptron_AsBaseEstimator.fit(x_train, y_train)
#     pkl_filename = "model_files/{}/split_{}/boosting_with_perceptron.pkl".format(portion, split_num)
#     with open(pkl_filename, 'wb') as file:
#         pickle.dump(model, file)
#
#     #boosting_classifiers_perceptron_AsBaseEstimator.fit(x_train, y_train)
#
#     #oracle_trained_model = boosting_classifiers_perceptron_AsBaseEstimator.fit(x_train, y_train)
#     #oracle_accuracy = oracle(x_train, y_train, model_name, oracle_trained_model)
#
#     y_pred = boosting_classifiers_perceptron_AsBaseEstimator.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
#     # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
#     return accuracy
#
# def random_forest(x_train, y_train, model_name):
#     random_forest_model = RandomForestClassifier(random_state=42)
#
#     random_forest_model.fit(x_train, y_train)
#
#     split_num = model_name.split('_')[4].split('.')[0]
#     portion = model_name.split('\\')[1]
#     model = random_forest_model.fit(x_train, y_train)
#     pkl_filename = "model_files/{}/split_{}/random_forest.pkl".format(portion, split_num)
#     with open(pkl_filename, 'wb') as file:
#         pickle.dump(model, file)
#
#
#
#     oracle_trained_model = random_forest_model.fit(x_train, y_train)
#     oracle_accuracy = oracle(x_train, y_train, model_name, oracle_trained_model)
#
#     y_pred = random_forest_model.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
#     return accuracy, oracle_accuracy

for files in glob.glob('preprocessed_files/*'):


    for items in glob.glob("{}/*.*".format(files)):
        data = np.load(items, allow_pickle=True)
        input_list = data.tolist()
        x_test = input_list['FrameStack'][1]
        y_test = input_list['FrameStack'][3]

        file_name = files.split("\\")[1]
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        model_load_1 = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        model_load_2 = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        model_load_3 = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        model_load_4 = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        model_load_5 = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]

        with open(model_load_1, 'rb') as file:
            my_model_1 = pickle.load(file)
        with open(model_load_2, 'rb') as file:
            my_model_2 = pickle.load(file)
        with open(model_load_3, 'rb') as file:
            my_model_3 = pickle.load(file)
        with open(model_load_4, 'rb') as file:
            my_model_4 = pickle.load(file)
        with open(model_load_5, 'rb') as file:
            my_model_5 = pickle.load(file)

        y_pred_1 = my_model_1.predict(x_test)
        y_pred_2 = my_model_2.predict(x_test)
        y_pred_3 = my_model_3.predict(x_test)
        y_pred_4 = my_model_4.predict(x_test)
        y_pred_5 = my_model_5.predict(x_test)


        y_pred_list = []

        y_pred_list.append(y_pred_1)
        y_pred_list.append(y_pred_2)
        y_pred_list.append(y_pred_3)
        y_pred_list.append(y_pred_4)
        y_pred_list.append(y_pred_5)

        total_preds = []
        for first_preds in y_pred_list:
            for second_preds in y_pred_list:
                # result = deslib.util.diversity.double_fault(y_test, first_preds, second_preds)
                # result = deslib.util.diversity.ratio_errors(y_test, first_preds, second_preds)
                result = deslib.util.diversity.Q_statistic(y_test, first_preds, second_preds)
                total_preds.append(result)

        print(items)
        # print(mean(total_preds))
    print(total_preds)
    print(mean(total_preds))

