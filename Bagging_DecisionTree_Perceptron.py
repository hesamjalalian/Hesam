import pickle
import glob
from numpy import mean
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

def bagging_with_decision_tree(x_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    Bagging_classifiers_DecisionTree_AsBaseEstimator = BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1, random_state=42)
    Bagging_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    y_pred = Bagging_classifiers_DecisionTree_AsBaseEstimator.predict(x_test) # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred) # Evaluate the accuracy
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def bagging_with_perceptron(x_train, y_train):
    perc = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
    Bagging_classifiers_Perceptron_AsBaseEstimator = BaggingClassifier(base_estimator=perc, n_estimators=100, n_jobs=-1, random_state=42)  # Instantiate a BaggingClassifier 'Bagging_classifiers_Perceptron_AsBaseEstimator'
    Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)  # Fit 'bc' to the training set
    y_pred = Bagging_classifiers_Perceptron_AsBaseEstimator.predict(x_test)  # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def boosting_with_decision_tree(x_train, y_train):
    bdt = DecisionTreeClassifier(random_state=42)
    boosting_classifiers_DecisionTree_AsBaseEstimator = AdaBoostClassifier(base_estimator=bdt, n_estimators=100, random_state=0, algorithm='SAMME')  # Instantiate a BoostingClassifier 'Boosting_classifiers_Perceptron_AsBaseEstimator'

    boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    y_pred = boosting_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

# def Boosting_with_perceptron(x_train, y_train):
#     bperc = Perceptron(tol=1e-3, random_state=42)
#     Boosting_classifiers_Perceptron_AsBaseEstimator = AdaBoostClassifier(base_estimator=bperc, n_estimators=100, random_state=0, algorithm='SAMME')
#     Boosting_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)
#     y_pred = Boosting_classifiers_Perceptron_AsBaseEstimator.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
#     # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
#     return accuracy


def save_obj(obj, name ):
    with open('{}.pkl'.format(name), 'wb') as configFile:
        pickle.dump(obj, configFile)

main_dir = glob.glob('preprocessed_files/*')

final_results_decisiontree = []
final_results_perceptron = []

# dictionary bejaye list
dict_final_results_Bagging_decision_tree = {}
dict_final_results_Bagging_perceptron = {}
dict_final_results_Boosting_decision_tree = {}
# dict_final_results_Boosting_perceptron = {}

for files in main_dir:
    # print(files)
    decisionTreeAccutrcies=[]
    perceptronAccurecies =[]
    boostingdecisionTreeAccutrcies = []
    # BoostingperceptronAccurecies = []
    for items in glob.glob("{}/*.*".format(files)):
        # print(items)

        data = np.load(items, allow_pickle=True)
        # print(data)
        input_list = data.tolist()   # How to get x_train
        x_train = input_list['FrameStack'][0]
        x_test = input_list['FrameStack'][1]
        y_train = input_list['FrameStack'][2]
        y_test = input_list['FrameStack'][3]
        # print(x_train)

        # print('Shape of X_train=>', x_train.shape)
        # print('Shape of X_test=>', x_test.shape)
        # print('Shape of Y_train=>', y_train.shape)
        # print('Shape of Y_test=>', y_test.shape)

        accuracy_bagging_with_decision_tree = bagging_with_decision_tree(x_train, y_train)
        accuracy_bagging_with_perceptron = bagging_with_perceptron(x_train, y_train)

        accuracy_Boosting_with_decision_tree = boosting_with_decision_tree(x_train, y_train)
        # accuracy_Boosting_with_perceptron = Boosting_with_perceptron(x_train, y_train)

        decisionTreeAccutrcies.append(accuracy_bagging_with_decision_tree)  #add accuracies to list
        perceptronAccurecies.append(accuracy_bagging_with_perceptron)

        boostingdecisionTreeAccutrcies.append(accuracy_Boosting_with_decision_tree) # add accuracies for boosting
        # BoostingperceptronAccurecies.append(accuracy_Boosting_with_perceptron)

    # print("file name : {}, decision tree mean accurecy : {}".format(files,mean(decisionTreeAccutrcies)))
    # print("file name : {}, perceptron mean accurecy    : {}".format(files,mean(perceptronAccurecies)))

    # final_results_decisiontree.append(mean(decisionTreeAccutrcies))
    # final_results_perceptron.append(mean(perceptronAccurecies))

    dict_final_results_Bagging_decision_tree["{}".format(files)] = mean(decisionTreeAccutrcies)
    dict_final_results_Bagging_perceptron["{}".format(files)] = mean(perceptronAccurecies)
    dict_final_results_Boosting_decision_tree["{}".format(files)] = mean(boostingdecisionTreeAccutrcies)
    # dict_final_results_Boosting_perceptron["{}".format(files)] = mean(BoostingperceptronAccurecies)

# print(dict_final_results_Bagging_decision_tree)
# print(dict_final_results_Bagging_perceptron)
# print(dict_final_results_Boosting_decision_tree)
# print(dict_final_results_Boosting_perceptron)

#save file
save_obj(dict_final_results_Bagging_decision_tree,"final_results_bagging_decisiontree")
save_obj(dict_final_results_Bagging_perceptron,"final_results_bagging_perceptron")
save_obj(dict_final_results_Boosting_decision_tree,"final_results_boosting_decisiontree")
# save_obj(dict_final_results_Boosting_perceptron,"final_results_boosting_decisiontree")





