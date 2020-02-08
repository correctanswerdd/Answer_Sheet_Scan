import cv2
import numpy as np
import os
import pickle


def split(data_dir="img/", train_prob=0.8):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    threshold = 100 * train_prob

    # split
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for i in files:
            type = np.random.randint(low=1, high=100, size=1, dtype=int)
            if type <= threshold:  # train
                X_train.append(cv2.imread(root + "/" + i))
                if i[-5] == 'A':
                    y_train.append([1, 0, 0])
                elif i[-5] == 'B':
                    y_train.append([0, 1, 0])
                elif i[-5] == 'C':
                    y_train.append([0, 0, 1])
                else:
                    pass
            else:
                X_test.append(cv2.imread(root + "/" + i))
                if i[-5] == 'A':
                    y_test.append([1, 0, 0])
                elif i[-5] == 'B':
                    y_test.append([0, 1, 0])
                elif i[-5] == 'C':
                    y_test.append([0, 0, 1])
                else:
                    pass

    # to nd.array
    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # save
    dir = 'data/'
    with open(dir + "X_train.pkl", 'wb') as f:
        pickle.dump(X_train, f)
    with open(dir + "y_train.pkl", 'wb') as f:
        pickle.dump(y_train, f)
    with open(dir + "X_test.pkl", 'wb') as f:
        pickle.dump(X_test, f)
    with open(dir + "Y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)


split()