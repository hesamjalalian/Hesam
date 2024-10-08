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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array



def random_forest(x_train, y_train, model_name):
    random_forest_model = RandomForestClassifier(random_state=42)

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = random_forest_model.fit(x_train, y_train)
    pkl_filename = "model_files/{}/split_{}/random_forest.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = random_forest_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def bagging_with_decision_tree(x_train, y_train, model_name):
    dt = DecisionTreeClassifier(max_depth=None)
    Bagging_classifiers_DecisionTree_AsBaseEstimator = BaggingClassifier(base_estimator=dt,n_estimators=100)

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = Bagging_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def bagging_with_perceptron(x_train, y_train, model_name):
    perc = Perceptron(shuffle=True)
    Bagging_classifiers_Perceptron_AsBaseEstimator = BaggingClassifier(base_estimator=perc,n_estimators=100)  # Instantiate a BaggingClassifier 'Bagging_classifiers_Perceptron_AsBaseEstimator'

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "model_files/{}/split_{}/bagging_with_perceptron.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = Bagging_classifiers_Perceptron_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def boosting_with_decision_tree(x_train, y_train, model_name):
    bdt = DecisionTreeClassifier(random_state=42, max_depth=1)
    boosting_classifiers_DecisionTree_AsBaseEstimator = AdaBoostClassifier(base_estimator=bdt, n_estimators=100,
                                                                           random_state=0,
                                                                           algorithm='SAMME')

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = boosting_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def Boosting_with_perceptron(x_train, y_train, model_name):
    bperc = Perceptron(tol=1e-3, random_state=42)
    Boosting_classifiers_perceptron_AsBaseEstimator = AdaBoostClassifier(base_estimator=bperc, n_estimators=100,
                                                                         random_state=0, algorithm='SAMME')

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Boosting_classifiers_perceptron_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "model_files/{}/split_{}/boosting_with_perceptron.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = Boosting_classifiers_perceptron_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def save_obj(obj, name):
    with open('{}.pkl'.format(name), 'wb') as configFile:
        pickle.dump(obj, configFile)


dict_final_results_Bagging_decision_tree = {}
dict_final_results_Bagging_perceptron = {}
dict_final_results_Boosting_decision_tree = {}
dict_final_results_random_forest = {}
dict_final_results_Boosting_perceptron = {}

dict_standard_deviation_decisionTreeAccutrcies = {}
dict_standard_deviation_perceptronAccurecies = {}
dict_standard_deviation_boostingdecisionTreeAccutrcies = {}
dict_standard_deviation_randomforestAccutrcies = {}
dict_standard_deviation_boostingperceptronAccutrcies = {}

for files in glob.glob('processed_3/*'):
    print(files)
    decisionTreeAccutrcies = []
    perceptronAccurecies = []
    boostingdecisionTreeAccutrcies = []
    randomforestAccutrcies = []
    boostingperceptronAccutrcies = []

    standard_deviation_decisionTreeAccutrcies = []
    standard_deviation_perceptronAccurecies = []
    standard_deviation_boostingdecisionTreeAccutrcies = []
    standard_deviation_randomforestAccutrcies = []
    standard_deviation_boostingperceptronAccutrcies = []

    for items in glob.glob("{}/*.*".format(files)):
        print(items)
        # print(len(items))

        data = np.load(items, allow_pickle=True)
        # print(data)
        input_list = data.tolist()  # How to get x_train
        x_train = input_list['FrameStack'][0]
        x_test = input_list['FrameStack'][1]
        y_train = input_list['FrameStack'][2]
        y_test = input_list['FrameStack'][3]
        x_dsel = input_list['FrameStack'][4]
        y_dsel = input_list['FrameStack'][5]
        print(len(x_train))
        print(len(x_test))
        print(len(x_dsel))
        print(len(y_train))
        print(len(y_test))
        print(len(y_dsel))


        # x_trainnn, x_validation, y_trainnn, y_validation = train_test_split(x_train, y_train, test_size=0.33,random_state=42)
        # print(len(x_trainnn))
        # print(len(x_validation))
        # print(len(y_trainnn))
        # print(len(y_validation))

        # print('Shape of X_train=>', x_train.shape)
        # print('Shape of X_test=>', x_test.shape)
        # print('Shape of Y_train=>', y_train.shape)
        # print('Shape of Y_test=>', y_test.shape)

        accuracy_bagging_with_decision_tree = bagging_with_decision_tree(x_train, y_train, items)
        # accuracy_bagging_with_perceptron = bagging_with_perceptron(x_train, y_train, items)
        # accuracy_Boosting_with_decision_tree = boosting_with_decision_tree(x_train, y_train, items)
        # accuracy_random_forest = random_forest(x_train, y_train, items)
        # accuracy_Boosting_with_perceptron = Boosting_with_perceptron(x_train, y_train, items)

        decisionTreeAccutrcies.append(accuracy_bagging_with_decision_tree)
        # perceptronAccurecies.append(accuracy_bagging_with_perceptron)
        # boostingdecisionTreeAccutrcies.append(accuracy_Boosting_with_decision_tree)
        # randomforestAccutrcies.append(accuracy_random_forest)
        # boostingperceptronAccutrcies.append(accuracy_Boosting_with_perceptron)

        # print(decisionTreeAccutrcies)
        # meannn = mean(decisionTreeAccutrcies)
        # print(meannn)
       
    dict_final_results_Bagging_decision_tree["{}".format(files)] = mean(decisionTreeAccutrcies)
    # dict_final_results_Bagging_perceptron["{}".format(files)] = mean(perceptronAccurecies)
    # dict_final_results_Boosting_decision_tree["{}".format(files)] = mean(boostingdecisionTreeAccutrcies)
    # dict_final_results_random_forest["{}".format(files)] = mean(randomforestAccutrcies)
    # dict_final_results_Boosting_perceptron["{}".format(files)] = mean(boostingperceptronAccutrcies)

    np.std(decisionTreeAccutrcies, dtype=np.float64)
    # np.std(perceptronAccurecies, dtype=np.float64)
    # np.std(boostingdecisionTreeAccutrcies, dtype=np.float64)
    # np.std(randomforestAccutrcies, dtype=np.float64)
    # np.std(boostingperceptronAccutrcies, dtype=np.float64)

    standard_deviation_decisionTreeAccutrcies.append(np.std(decisionTreeAccutrcies, dtype=np.float64))
    # standard_deviation_perceptronAccurecies.append(np.std(perceptronAccurecies, dtype=np.float64))
    # standard_deviation_boostingdecisionTreeAccutrcies.append(np.std(boostingdecisionTreeAccutrcies, dtype=np.float64))
    # standard_deviation_randomforestAccutrcies.append(np.std(randomforestAccutrcies, dtype=np.float64))
    # standard_deviation_boostingperceptronAccutrcies.append(np.std(boostingperceptronAccutrcies, dtype=np.float64))
    # print(standard_deviation_decisionTreeAccutrcies)

    dict_standard_deviation_decisionTreeAccutrcies["{}".format(files)] = mean(standard_deviation_decisionTreeAccutrcies)
    # dict_standard_deviation_perceptronAccurecies["{}".format(files)] = mean(standard_deviation_perceptronAccurecies)
    # dict_standard_deviation_boostingdecisionTreeAccutrcies["{}".format(files)] = mean(standard_deviation_boostingdecisionTreeAccutrcies)
    # dict_standard_deviation_randomforestAccutrcies["{}".format(files)] = mean(standard_deviation_randomforestAccutrcies)
    # dict_standard_deviation_boostingperceptronAccutrcies["{}".format(files)] = mean(standard_deviation_boostingperceptronAccutrcies)

print(dict_final_results_Bagging_decision_tree)
# print(dict_final_results_Bagging_perceptron)
# print(dict_final_results_Boosting_decision_tree)
# print(dict_final_results_random_forest)
# print(dict_final_results_Boosting_perceptron)

print(dict_final_results_Bagging_decision_tree.values())
# print(dict_final_results_Bagging_perceptron.values())
# print(dict_final_results_Boosting_decision_tree.values())
# print(dict_final_results_random_forest.values())
# print(dict_final_results_Boosting_perceptron.values())

print(dict_standard_deviation_decisionTreeAccutrcies.values())
# print(dict_standard_deviation_perceptronAccurecies.values())
# print(dict_standard_deviation_boostingdecisionTreeAccutrcies.values())
# print(dict_standard_deviation_randomforestAccutrcies.values())
# print(dict_standard_deviation_boostingperceptronAccutrcies.values())

# #save file
# save_obj(dict_final_results_Bagging_decision_tree, "final_results_bagging_decisiontree")
# save_obj(dict_final_results_Bagging_perceptron, "final_results_bagging_perceptron")
# save_obj(dict_final_results_Boosting_decision_tree, "final_results_boosting_decisiontree")
# save_obj(dict_final_results_random_forest, "final_results_randomforest")
# save_obj(dict_final_results_Boosting_perceptron, "final_results_boosting_perceptron")

Bagging_decision_tree = (dict_final_results_Bagging_decision_tree.values())
# Bagging_perceptron = (dict_final_results_Bagging_perceptron.values())
# Boosting_decision_tree = (dict_final_results_Boosting_decision_tree.values())
# random_forest = (dict_final_results_random_forest.values())
# Boosting_perceptron = (dict_final_results_Boosting_perceptron.values())

standard_deviation_bagging_decisionTree = (dict_standard_deviation_decisionTreeAccutrcies.values())
# standard_deviation_Bagging_perceptron = (dict_standard_deviation_perceptronAccurecies.values())
# standard_deviation_boosting_decisionTree = (dict_standard_deviation_boostingdecisionTreeAccutrcies.values())
# standard_deviation_randomforest = (dict_standard_deviation_randomforestAccutrcies.values())
# standard_deviation_boosting_perceptron = (dict_standard_deviation_boostingperceptronAccutrcies.values())
#
