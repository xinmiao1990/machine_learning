import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm


class LogisticCLF():
    def __init__(self, n_classes, step_size=0.5, max_iter=1000, e=0.0001):
        self.n_classes = n_classes
        self.step_size = step_size
        self.max_iter = max_iter
        self.e = e

    def train(self, X, y,
                         w0=None,
                         b0=None):
        """
        Inputs:
        - X: training features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features
        - y: multiclass training labels, a N dimensional numpy array where
        N is the number of training points, indicating the labels of
        training data
        - C: number of classes in the data
        - step_size: step size (learning rate)
        - max_iter: maximum number for iterations to perform

        Returns:
        - w: C-by-D weight matrix of multinomial logistic regression, where
        C is the number of classes and D is the dimensionality of features.
        - b: bias vector of length C, where C is the number of classes

        Implement a multinomial logistic regression for multiclass
        classification. Keep in mind, that for this task you may need a
        special (one-hot) representation of classification labels, where
        each label y_i is represented as a row of zeros with a single 1 in
        the column, that corresponds to the class y_i belongs to.
        """
        C = self.n_classes
        N, D = X.shape

        w = np.zeros((C, D))
        if w0 is not None:
            w = w0

        b = np.zeros(C)
        if b0 is not None:
            b = b0

        wpb = np.zeros((C, D+1))
        wpb[:, 0:D] = w
        wpb[:, D] = b
        Xp1 = np.concatenate([X, np.ones((N, 1))], axis=1)

        for j in range(0, self.max_iter):
            wpb_p = np.copy(wpb)
            temp = self.softmax(Xp1, wpb_p)

            for i in range(0, C):
                wpb_i = wpb[i, :]
                y_i = 1 * (y == i)

                g = np.dot(Xp1.T,  temp[i, :] - y_i) / N
                assert g.shape == ((D + 1),)
                wpb_i = wpb_i - self.step_size * g
                wpb[i, :] = wpb_i

            max_relative_diff_w = np.amax(np.abs(wpb_p - wpb) / (np.abs(wpb_p) + 1e-8))
            tmp = self.softmax(Xp1, wpb)
            test = np.argmax(tmp, axis=0)
            error = np.sum(1 * (test == y))
            if max_relative_diff_w < self.e:
                break
            print('Iter: %d,  Error: %f\n' % (j, float(N - error)/N ) )

        w = wpb[:, 0:D]
        b = wpb[:, D]

        assert w.shape == (C, D)
        assert b.shape == (C,)
        return w, b

    def predict(self, X, w, b):
        """
        Inputs:
        - X: testing features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features
        - w: weights of the trained multinomial classifier
        - b: bias terms of the trained multinomial classifier

        Returns:
        - preds: N dimensional vector of multiclass predictions.
        Outputted predictions should be from {0, C - 1}, where
        C is the number of classes

        Make predictions for multinomial classifier.
        """
        assert X.shape[1] == w.shape[1]
        N, D = X.shape
        C = self.n_classes
        preds = np.zeros(N)

        wpb = np.zeros((C, D + 1))
        wpb[:, 0:D] = w
        wpb[:, D] = b
        Xp1 = np.concatenate([X, np.ones((N, 1))], axis=1)

        y = self.softmax(Xp1, wpb)
        preds = np.argmax(y, axis=0)

        assert preds.shape == (N,)
        return preds

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, X, w):
        assert X.shape[1] == w.shape[1]
        N = X.shape[0]
        C = self.n_classes
        denom = np.exp(np.dot(w, X.T))
        assert denom.shape == (C, N)

        return denom / np.sum(denom, axis=0)

    def accuracy_score(self, true_val, preds):
        return np.sum(true_val == preds).astype(float) / len(true_val)
