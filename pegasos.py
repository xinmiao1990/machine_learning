import json
import numpy as np


class SVM():
    def __init__(self, num_classes, k=100, max_iterations=100, lamb=0.1, t=0.):
        self.num_classes = num_classes
        self.k = k
        self.max_iterations = max_iterations
        self.lamb = lamb
        self.t = t

    def objective_function(self, X, y, w):
        """
        Inputs:
        - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
        - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
        - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
        - lamb: lambda used in pegasos algorithm

        Return:
        - train_obj: the value of objective function in SVM primal formulation
        """
        N = X.shape[0]
        zero_vec = np.array([0]*N)
        obj_value = self.lamb/2 * np.dot(w.T, w) + 1/N * np.sum(np.maximum(1 - y * np.dot(X, w), zero_vec))

        return obj_value

    def binary_train(self, Xtrain, ytrain):
        """
        Inputs:
        - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
        - ytrain: A list of num_train labels
        - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
        - lamb: lambda used in pegasos algorithm
        - k: mini-batch size
        - max_iterations: the maximum number of iterations to update parameters

        Returns:
        - learnt w
        - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
        """
        np.random.seed(0)
        N = Xtrain.shape[0]
        D = Xtrain.shape[1]

        for iter in range(1, self.max_iterations + 1):
            A_t = np.floor(np.random.rand(self.k) * N).astype(int)  # index of the current mini-batch
            if iter == 1:
                w = np.array([0.0]*D)
            else:
                wtp = w

                eta_t = 1/self.lamb/iter
                X_batch = Xtrain[A_t, :]
                y_batch = ytrain[A_t]

                yXw = y_batch * np.dot(X_batch, w)
                # assert yXw.shape == (k, )
                Atp = (yXw < 1.0)
                # X_batch.T: D * k; y_batch: k
                yX = np.dot(X_batch[Atp, :].T, y_batch[Atp])
                # assert yX.shape == (D,)
                wt_half = (1 - eta_t * self.lamb) * wtp + eta_t/self.k * yX
                # assert w.shape == wt_half.shape

                scal = 1 / np.sqrt(self.lamb) / np.sqrt(np.dot(wt_half.T, wt_half))
                scal = np.minimum(1.0, scal)
                w = wt_half * scal

        return w

    def binary_predict(self, Xtest, w):
        """
        Inputs:
        - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
        - ytest: A list of num_test labels
        - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
        - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

        Returns:
        - test_acc: testing accuracy.
        """

        N = Xtest.shape[0]
        y_est = np.dot(Xtest, w)
        result_est = np.array([-1.0]*N)
        result_est[y_est > self.t] = 1.0

        return result_est

    def OVR_train(self, X, y):
        """
        Inputs:
        - X: training features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features
        - y: multiclass training labels, a N dimensional numpy array,
        indicating the labels of each training point
        - C: number of classes in the data
        - w0: initial value of weight matrix
        - b0: initial value of bias term
        - step_size: step size (learning rate)
        - max_iterations: maximum number of iterations for gradient descent

        Returns:
        - w: a C-by-D weight matrix of OVR logistic regression
        - b: bias vector of length C

        Implement multiclass classification using binary classifier and
        one-versus-rest strategy. Recall, that the OVR classifier is
        trained by training C different classifiers.
        """
        N, D = X.shape

        w_final = np.zeros((self.num_classes, D))

        for i in range(0, self.num_classes):
            y_i = 1 * (y == i) + (-1) * (y != i)
            w_i = self.binary_train(X, y_i)
            w_final[i, :] = w_i
        return w_final

    def OVR_predict(self, X, w):
        """
        Inputs:
        - X: testing features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features
        - w: weights of the trained OVR model
        - b: bias terms of the trained OVR model

        Returns:
        - preds: vector of class label predictions.
        Outputted predictions should be from {0, C - 1}, where
        C is the number of classes.

        Make predictions using OVR strategy and predictions from binary
        classifier.
        """
        N, D = X.shape

        y = np.zeros((self.num_classes, N))
        for i in range(0, self.num_classes):
            w_i = w[i, :]
            y_i = self.binary_predict(X, w_i)
            y[i, :] = y_i

        preds = np.argmax(y, axis=0)
        return preds

