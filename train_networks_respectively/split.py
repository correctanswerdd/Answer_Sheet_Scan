import cv2
import numpy as np
import os
import pickle


def split(data_dir="img_re/", train_prob=0.8, val_prob=0.1, shuffle=2):
    X = []
    y = []

    # to list
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for i in files:
            if i[-5] == 'A':
                X.append(cv2.imread(root + "/" + i))
                y.append([1, 0, 0, 0])
            elif i[-5] == 'B':
                X.append(cv2.imread(root + "/" + i))
                y.append([0, 1, 0, 0])
            elif i[-5] == 'C':
                X.append(cv2.imread(root + "/" + i))
                y.append([0, 0, 1, 0])
            elif i[-5] == 'D':
                X.append(cv2.imread(root + "/" + i))
                y.append([0, 0, 0, 1])

    # to array
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    # shuffle
    index = [i for i in range(X.shape[0])]
    for k in range(shuffle):
        np.random.shuffle(index)
    X = X[index, :]
    y = y[index, :]

    # split
    threshold_1 = int(X.shape[0] * train_prob)
    threshold_2 = int(X.shape[0] * (1 - val_prob))
    index_train = [i for i in range(threshold_1)]
    index_val = [(i + threshold_1) for i in range(threshold_2 - threshold_1)]
    index_test = [(i + threshold_2) for i in range(X.shape[0] - threshold_2)]
    X_train = X[index_train, :]
    y_train = y[index_train, :]
    X_test = X[index_test, :]
    y_test = y[index_test, :]
    X_val = X[index_val, :]
    y_val = y[index_val, :]

    # save
    dir = 'data/'
    with open(dir + "X_train.pkl", 'wb') as f:
        pickle.dump(X_train, f)
    with open(dir + "y_train.pkl", 'wb') as f:
        pickle.dump(y_train, f)
    with open(dir + "X_test.pkl", 'wb') as f:
        pickle.dump(X_test, f)
    with open(dir + "y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)
    with open(dir + "X_val.pkl", 'wb') as f:
        pickle.dump(X_val, f)
    with open(dir + "Y_val.pkl", 'wb') as f:
        pickle.dump(y_val, f)


split()