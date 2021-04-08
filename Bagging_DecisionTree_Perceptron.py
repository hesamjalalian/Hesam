import glob
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

def bagging_with_decision_tree(x_train, y_train, model_name):
    dt = DecisionTreeClassifier(random_state=42)
    Bagging_classifiers_DecisionTree_AsBaseEstimator = BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1, random_state=42)

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_DecisionTree_AsBaseEstimator
    pkl_filename = "model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    Bagging_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    y_pred = Bagging_classifiers_DecisionTree_AsBaseEstimator.predict(x_test) # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred) # Evaluate the accuracy
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def bagging_with_perceptron(x_train, y_train, model_name):
    perc = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
    Bagging_classifiers_Perceptron_AsBaseEstimator = BaggingClassifier(base_estimator=perc, n_estimators=100, n_jobs=-1, random_state=42)  # Instantiate a BaggingClassifier 'Bagging_classifiers_Perceptron_AsBaseEstimator'

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_Perceptron_AsBaseEstimator
    pkl_filename = "model_files/{}/split_{}/bagging_with_perceptron.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    Bagging_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)  # Fit 'bc' to the training set
    y_pred = Bagging_classifiers_Perceptron_AsBaseEstimator.predict(x_test)  # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def boosting_with_decision_tree(x_train, y_train, model_name):
    bdt = DecisionTreeClassifier(random_state=42)
    boosting_classifiers_DecisionTree_AsBaseEstimator = AdaBoostClassifier(base_estimator=bdt, n_estimators=100, random_state=0, algorithm='SAMME')  # Instantiate a BoostingClassifier 'Boosting_classifiers_Perceptron_AsBaseEstimator'

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = boosting_classifiers_DecisionTree_AsBaseEstimator
    pkl_filename = "model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    y_pred = boosting_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def Boosting_with_perceptron(x_train, y_train):
    bperc = Perceptron(tol=1e-3, random_state=42)
    Boosting_classifiers_Perceptron_AsBaseEstimator = AdaBoostClassifier(base_estimator=bperc, n_estimators=100, random_state=0, algorithm='SAMME')
    Boosting_classifiers_Perceptron_AsBaseEstimator.fit(x_train, y_train)
    y_pred = Boosting_classifiers_Perceptron_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def bagging_with_random_forest(x_train, y_train, model_name):
    rf = RandomForestClassifier(random_state=42)
    Bagging_classifiers_randomforest_AsBaseEstimator = BaggingClassifier(base_estimator=rf, n_estimators=100, n_jobs=-1, random_state=42)

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_randomforest_AsBaseEstimator
    pkl_filename = "model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    Bagging_classifiers_randomforest_AsBaseEstimator.fit(x_train, y_train)
    y_pred = Bagging_classifiers_randomforest_AsBaseEstimator.predict(x_test) # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred) # Evaluate the accuracy
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def boosting_with_random_forest(x_train, y_train, model_name):
    brf = RandomForestClassifier(random_state=42)
    boosting_classifiers_randomforest_AsBaseEstimator = AdaBoostClassifier(base_estimator=brf, n_estimators=100, random_state=0, algorithm='SAMME')  # Instantiate a BoostingClassifier 'Boosting_classifiers_Perceptron_AsBaseEstimator'

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = boosting_classifiers_randomforest_AsBaseEstimator
    pkl_filename = "model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    boosting_classifiers_randomforest_AsBaseEstimator.fit(x_train, y_train)
    y_pred = boosting_classifiers_randomforest_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def save_obj(obj, name ):
    with open('{}.pkl'.format(name), 'wb') as configFile:
        pickle.dump(obj, configFile)

# dictionary bejaye list
dict_final_results_Bagging_decision_tree = {}
dict_final_results_Bagging_perceptron = {}
dict_final_results_Boosting_decision_tree = {}
dict_final_results_Bagging_random_forest = {}
dict_final_results_Boosting_random_forest = {}
# dict_final_results_Boosting_perceptron = {}

dict_standard_deviation_decisionTreeAccutrcies = {}
dict_standard_deviation_perceptronAccurecies = {}
dict_standard_deviation_boostingdecisionTreeAccutrcies = {}
dict_standard_deviation_bagging_randomforestAccutrcies = {}
dict_standard_deviation_boosting_randomforestAccutrcies = {}

for files in glob.glob('preprocessed_files/*'):
    print(files)
    decisionTreeAccutrcies=[]
    perceptronAccurecies =[]
    boostingdecisionTreeAccutrcies = []
    baggingrandomforestAccutrcies = []
    boostingrandomforestAccutrcies = []

    standard_deviation_decisionTreeAccutrcies = []
    standard_deviation_perceptronAccurecies = []
    standard_deviation_boostingdecisionTreeAccutrcies = []
    standard_deviation_bagging_randomforestAccutrcies = []
    standard_deviation_boosting_randomforestAccutrcies =[]

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

        accuracy_bagging_with_decision_tree = bagging_with_decision_tree(x_train, y_train, items)
        accuracy_bagging_with_perceptron = bagging_with_perceptron(x_train, y_train, items)
        accuracy_Boosting_with_decision_tree = boosting_with_decision_tree(x_train, y_train, items)
        accuracy_bagging_with_random_forest = bagging_with_random_forest(x_train, y_train, items)
        accuracy_boosting_with_random_forest = boosting_with_random_forest(x_train, y_train, items)
        # accuracy_Boosting_with_perceptron = Boosting_with_perceptron(x_train, y_train)

        decisionTreeAccutrcies.append(accuracy_bagging_with_decision_tree)  #add accuracies to list
        perceptronAccurecies.append(accuracy_bagging_with_perceptron)
        boostingdecisionTreeAccutrcies.append(accuracy_Boosting_with_decision_tree) # add accuracies for boosting
        baggingrandomforestAccutrcies.append(accuracy_bagging_with_random_forest)
        boostingrandomforestAccutrcies.append(accuracy_boosting_with_random_forest)

        # BoostingperceptronAccurecies.append(accuracy_Boosting_with_perceptron)

    dict_final_results_Bagging_decision_tree["{}".format(files)] = mean(decisionTreeAccutrcies)
    dict_final_results_Bagging_perceptron["{}".format(files)] = mean(perceptronAccurecies)
    dict_final_results_Boosting_decision_tree["{}".format(files)] = mean(boostingdecisionTreeAccutrcies)
    dict_final_results_Bagging_random_forest["{}".format(files)] = mean(baggingrandomforestAccutrcies)
    dict_final_results_Boosting_random_forest["{}".format(files)] = mean(boostingrandomforestAccutrcies)
    # dict_final_results_Boosting_perceptron["{}".format(files)] = mean(BoostingperceptronAccurecies)

    np.std(decisionTreeAccutrcies, dtype=np.float64)
    np.std(perceptronAccurecies, dtype=np.float64)
    np.std(boostingdecisionTreeAccutrcies, dtype=np.float64)
    np.std(baggingrandomforestAccutrcies, dtype=np.float64)
    np.std(boostingrandomforestAccutrcies, dtype=np.float64)

    standard_deviation_decisionTreeAccutrcies.append(np.std(decisionTreeAccutrcies, dtype=np.float64))
    standard_deviation_perceptronAccurecies.append(np.std(perceptronAccurecies, dtype=np.float64))
    standard_deviation_boostingdecisionTreeAccutrcies.append(np.std(boostingdecisionTreeAccutrcies, dtype=np.float64))
    standard_deviation_bagging_randomforestAccutrcies.append(np.std(baggingrandomforestAccutrcies, dtype=np.float64))
    standard_deviation_boosting_randomforestAccutrcies.append(np.std(boostingrandomforestAccutrcies, dtype=np.float64))
    # print(standard_deviation_decisionTreeAccutrcies)

    dict_standard_deviation_decisionTreeAccutrcies["{}".format(files)] = mean(standard_deviation_decisionTreeAccutrcies)
    dict_standard_deviation_perceptronAccurecies["{}".format(files)] = mean(standard_deviation_perceptronAccurecies)
    dict_standard_deviation_boostingdecisionTreeAccutrcies["{}".format(files)] = mean(standard_deviation_boostingdecisionTreeAccutrcies)
    dict_standard_deviation_bagging_randomforestAccutrcies["{}".format(files)] = mean(standard_deviation_bagging_randomforestAccutrcies)
    dict_standard_deviation_boosting_randomforestAccutrcies["{}".format(files)] = mean(standard_deviation_boosting_randomforestAccutrcies)

print(dict_final_results_Bagging_decision_tree)
print(dict_final_results_Bagging_perceptron)
print(dict_final_results_Boosting_decision_tree)
print(dict_final_results_Bagging_random_forest)
print(dict_final_results_Boosting_random_forest)
# print(dict_final_results_Boosting_perceptron)

print(dict_final_results_Bagging_decision_tree.values())
print(dict_final_results_Bagging_perceptron.values())
print(dict_final_results_Boosting_decision_tree.values())
print(dict_final_results_Bagging_random_forest.values())
print(dict_final_results_Boosting_random_forest.values())

print(dict_standard_deviation_decisionTreeAccutrcies.values())
print(dict_standard_deviation_perceptronAccurecies.values())
print(dict_standard_deviation_boostingdecisionTreeAccutrcies.values())
print(dict_standard_deviation_bagging_randomforestAccutrcies.values())
print(dict_standard_deviation_boosting_randomforestAccutrcies.values())

#save file
save_obj(dict_final_results_Bagging_decision_tree,"final_results_bagging_decisiontree")
save_obj(dict_final_results_Bagging_perceptron,"final_results_bagging_perceptron")
save_obj(dict_final_results_Boosting_decision_tree,"final_results_boosting_decisiontree")
save_obj(dict_final_results_Bagging_random_forest,"final_results_bagging_randomforest")
save_obj(dict_final_results_Boosting_decision_tree,"final_results_boosting_randomforest")
# save_obj(dict_final_results_Boosting_perceptron,"final_results_boosting_decisiontree")

Bagging_decision_tree = (dict_final_results_Bagging_decision_tree.values())
Bagging_perceptron = (dict_final_results_Bagging_perceptron.values())
Boosting_decision_tree = (dict_final_results_Boosting_decision_tree.values())
Bagging_random_forest = (dict_final_results_Bagging_random_forest.values())
Boosting_random_forest = (dict_final_results_Boosting_random_forest.values())

standard_deviation_bagging_decisionTree = (dict_standard_deviation_decisionTreeAccutrcies.values())
standard_deviation_Bagging_perceptron = (dict_standard_deviation_perceptronAccurecies.values())
standard_deviation_boosting_decisionTree = (dict_standard_deviation_boostingdecisionTreeAccutrcies.values())
standard_deviation_bagging_randomforest = (dict_standard_deviation_bagging_randomforestAccutrcies.values())
standard_deviation_boosting_randomforest = (dict_standard_deviation_boosting_randomforestAccutrcies.values())

list_of_tuples = list(zip(Bagging_decision_tree,standard_deviation_bagging_decisionTree, Bagging_perceptron,standard_deviation_Bagging_perceptron, Boosting_decision_tree,standard_deviation_boosting_decisionTree ))
list_of_tuples
# df = pd.DataFrame(list_of_tuples,columns=['Bagging_decision_tree','standard_deviation_bagging_decisionTree', 'Bagging_perceptron', 'standard_deviation_Bagging_perceptron', 'Boosting_decision_tree', 'standard_deviation_boosting_decisionTree'])
# print(df)

# with open('table.txt', 'w') as f:
#     f.write(tabulate(df))
#
# with open('filename.txt', 'w') as outputfile:
#     print(tabulate(df), file=outputfile)

Point = make_dataclass("Point", [("Bagging_decision_tree", int),
                                 ("standard_deviation_bagging_decisionTree", int),
                                 ("Bagging_perceptron", int),
                                 ("standard_deviation_Bagging_perceptron", int),
                                 ("Boosting_decision_tree", int),
                                 ("standard_deviation_boosting_decisionTree", int)])
df = pd.DataFrame(list_of_tuples,columns=['Bagging_decision_tree','standard_deviation_bagging_decisionTree', 'Bagging_perceptron', 'standard_deviation_Bagging_perceptron', 'Boosting_decision_tree', 'standard_deviation_boosting_decisionTree'])

with open('table.txt', 'w') as f:
    f.write(tabulate(df))

with open('filename.txt', 'w') as outputfile:
    print(tabulate(df), file=outputfile)










