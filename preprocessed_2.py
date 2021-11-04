import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# file_list = glob.glob('datasets/*.*')

for file in glob.glob("dataset/*.*"):
# for file in glob.glob("datasets/*.*"):
    input = loadmat(file)
    data = input['dataset']

    X = data[:, 0:-1]
    y = data[:, -1]





    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=1000)
    sss.get_n_splits(X, y)
    split_part=0
    name = file.split(".")[0].split("\\")[1]

    for train_index, test_index in sss.split(X, y):
        split_part=split_part+1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_1, x_dsel, y_train_1, y_dsel = train_test_split(X_train, y_train, test_size=0.33,random_state=42)
        print(len(x_train_1))
        print(len(X_test))
        print(len(x_dsel))
        print(len(y_train_1))
        print(len(y_test))
        print(len(y_dsel))

        label_encoder = preprocessing.LabelEncoder()
        y_train_ms = label_encoder.fit_transform(y_train_1)
        y_test_ms = label_encoder.transform(y_test)
        y_dsel_ms = label_encoder.fit_transform(y_dsel)


        # normalization
        ms = MinMaxScaler()
        X_train_ms = ms.fit_transform(x_train_1)
        X_test_ms = ms.transform(X_test)
        X_dsel_ms = ms.fit_transform(x_dsel)


        # handling the missing data and replace missing values with mean of all the other values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(X_train_ms[:, 1:])
        imputer = imputer.fit(X_dsel_ms[:, 1:])
        X_train_ms[:, 1:] = imputer.transform(X_train_ms[:, 1:])
        X_test_ms[:, 1:] = imputer.transform(X_test_ms[:, 1:])
        X_dsel_ms[:, 1:] = imputer.transform(X_dsel_ms[:, 1:])

        X_train_processed = X_train_ms
        X_test_processed = X_test_ms
        X_dsel_processed = X_dsel_ms
        y_train_processed = y_train_ms
        y_test_processed = y_test_ms
        y_dsel_processed = y_dsel_ms

        processed = [X_train_processed, X_test_processed, y_train_processed, y_test_processed, X_dsel_processed, y_dsel_processed]
        # processed = [X_train_processed, y_train_processed]
        # print(file)
        name = file.split(".")[0].split("\\")[1]
        print(name)

        # How to save data in mat format
        FramStack = np.empty((len(processed),), dtype=object)
        for i in range(len(processed)):
            FramStack[i]=processed[i]



        # savemat("myfile.mat", {"FrameStack": FramStack})
        # savemat('processed_2/{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})
        # how to save data in numpy format
        # np.save('processed_2/{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack":FramStack})
        # np.save('processed_for_flt/{}_dir/processed_for_flt_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})

        # np.save('processed_4/{}_dir/processed_{}_{}'.format(name, name, split_part),{"FrameStack": FramStack})
        np.save('processed_3/{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})
        # np.save('{}_dir/processed_3_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})