import cv2
import numpy as np
import pickle


def shuffle(data_dir="data/", count=3):
    with open(data_dir + 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir + 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir + 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir + 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    # shuffle
    index_train = [i for i in range(X_train.shape[0])]
    index_test = [i for i in range(X_test.shape[0])]
    for k in range(count):
        np.random.shuffle(index_train)
        np.random.shuffle(index_test)
    X_train = X_train[index_train, :]
    y_train = y_train[index_train, :]
    X_test = X_test[index_test, :]
    y_test = y_test[index_test, :]

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


shuffle()