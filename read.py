import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import ShuffleSplit
import pickle
import cv2
import glob
from sklearn.model_selection import StratifiedShuffleSplit


dataset = loadmat(r"C:\Users\Hesam\27.2.2021\matAdult.mat")
print(dataset)