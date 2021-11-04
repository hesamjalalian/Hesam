from deslib.dcs.ola import OLA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import numpy as np
from numpy import mean
import pickle


dict_final_results = {}
dict_standard_deviation = {}

# for files in glob.glob('preprocessed_files/*'):
for files in glob.glob('processed_3/*'):
    print(files)

    standard_deviation = []
    accuracy = []
    ola_results = []

    for items in glob.glob("{}/*.*".format(files)):
        print(items)




        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        print(item_name)
        ss = file_name.split("_")[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        model_load = glob.glob("Models_FLT_10_2/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
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
            x_dsel = input_list['FrameStack'][4]
            y_dsel = input_list['FrameStack'][5]
            # print(x_train)
            # print(len(x_train))
            # print(len(x_test))


            # x_trainnn, x_dsel, y_trainnn, y_dsel = train_test_split(x_train, y_train, test_size=0.33,random_state=42)
            # print(len(x_trainnn))
            # print(len(x_dsel))
            # print(len(y_trainnn))
            # print(len(y_dsel))

            pool_classifiers = my_model



            # Initialize the DES model
            ola = OLA(pool_classifiers)

            # Preprocess the Dynamic Selection dataset (DSEL)
            ola.fit(x_dsel, y_dsel)

            # Predict new examples:
            yh = ola.predict(x_test)
            acc = accuracy_score(y_test, yh)

            ola_results.append(acc)
            print(acc)
    meannn = mean(ola_results)
    print(meannn)

    # dict_final_results["{}".format(files)] = mean(ola_results)
    # print(dict_final_results)
#     np.std(ola_results, dtype=np.float64)
#     standard_deviation.append(np.std(ola_results, dtype=np.float64))
#     dict_standard_deviation["{}".format(files)] = mean(standard_deviation)
#     # print(dict_standard_deviation.values())
#
#
# print(dict_standard_deviation.values())
# standard_deviation = (dict_standard_deviation.values())
#
