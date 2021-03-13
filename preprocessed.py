import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from sklearn.model_selection import StratifiedShuffleSplit


file_list = glob.glob('datasets/*.*')

for file in glob.glob("datasets/*.*"):
    input = loadmat(file)
    data = input['dataset']

    X = data[:, 0:-1]
    y = data[:, -1]

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=0)
    sss.get_n_splits(X, y)
    split_part=0
    name = file.split(".")[0].split("\\")[1]

    for train_index, test_index in sss.split(X, y):
        split_part=split_part+1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        label_encoder = preprocessing.LabelEncoder()
        y_train_ms = label_encoder.fit_transform(y_train)
        y_test_ms = label_encoder.transform(y_test)

        # normalization
        ms = MinMaxScaler()
        X_train_ms = ms.fit_transform(X_train)
        X_test_ms = ms.transform(X_test)


        # handling the missing data and replace missing values with mean of all the other values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(X_train_ms[:, 1:])
        X_train_ms[:, 1:] = imputer.transform(X_train_ms[:, 1:])
        X_test_ms[:, 1:] = imputer.transform(X_test_ms[:, 1:])

        X_train_processed = X_train_ms
        X_test_processed = X_test_ms
        y_train_processed = y_train_ms
        y_test_processed = y_test_ms

        processed = [X_train_processed, X_test_processed, y_train_processed, y_test_processed]
        #print(file)
        name = file.split(".")[0].split("\\")[1]
        print(name)

        # How to save data in mat format
        FramStack = np.empty((len(processed),), dtype=object)
        for i in range(len(processed)):
            FramStack[i]=processed[i]
        # how to save data in numpy format
        np.save('{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack":FramStack})




