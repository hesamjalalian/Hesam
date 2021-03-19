import pickle
import glob
from numpy import mean
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

def bagging_with_decision_tree(x_train, y_train):
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.3, random_state=1)
    bc = BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1)
    bc.fit(x_train, y_train)
    y_pred = bc.predict(x_test) # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred) # Evaluate the accuracy
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def bagging_with_perceptron(x_train, y_train):
    perc = Perceptron(tol=1e-3, random_state=0)
    bc = BaggingClassifier(base_estimator=perc, n_estimators=100, n_jobs=-1)  # Instantiate a BaggingClassifier 'bc'
    bc.fit(x_train, y_train)  # Fit 'bc' to the training set
    y_pred = bc.predict(x_test)  # Predict test set labels
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate the accuracy
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

def save_obj(obj, name ):
    with open('{}.pkl'.format(name), 'wb') as configFile:
        pickle.dump(obj, configFile)

main_dir = glob.glob('preprocessed_files/*')

final_results_decisiontree = []
final_results_perceptron = []

# dictionary bejaye list
dict_final_results_decisiontree = {}
dict_final_results_perceptron = {}

for files in main_dir:
    # print(files)
    decisionTreeAccutrcies=[]
    perceptronAccurecies =[]
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
        accuracy_bagging_with_perceptron =bagging_with_perceptron(x_train, y_train)

        decisionTreeAccutrcies.append(accuracy_bagging_with_decision_tree)  #add accuracies to list
        perceptronAccurecies.append(accuracy_bagging_with_perceptron)
        # print(accuracy)

    # print("file name : {}, decision tree mean accurecy : {}".format(files,mean(decisionTreeAccutrcies)))
    # print("file name : {}, perceptron mean accurecy    : {}".format(files,mean(perceptronAccurecies)))

    # final_results_decisiontree.append(mean(decisionTreeAccutrcies))
    # final_results_perceptron.append(mean(perceptronAccurecies))

    dict_final_results_decisiontree["{}".format(files)] = mean(decisionTreeAccutrcies)
    dict_final_results_perceptron["{}".format(files)] = mean(perceptronAccurecies)

print(dict_final_results_decisiontree)
print(dict_final_results_perceptron)
# save file as pickle
save_obj(dict_final_results_perceptron,"final_results_perceptron")
save_obj(dict_final_results_decisiontree,"final_results_decisiontree")





