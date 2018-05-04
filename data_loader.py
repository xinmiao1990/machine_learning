import json
import numpy as np


def data_loader_mnist(dataset='mnist_subset.json'):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    Ytrain = train_set[1]
    Xvalid = valid_set[0]
    Yvalid = valid_set[1]
    Xtest = test_set[0]
    Ytest = test_set[1]

    return np.array(Xtrain), np.array(Ytrain), np.array(Xvalid), \
           np.array(Yvalid), np.array(Xtest), np.array(Ytest)

    # return np.array(Xtrain).reshape(-1, 1, 28, 28), np.array(Ytrain), np.array(Xvalid).reshape(-1, 1, 28, 28),\
    #        np.array(Yvalid), np.array(Xtest).reshape(-1, 1, 28, 28), np.array(Ytest)
