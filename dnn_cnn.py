import json
import numpy as np
import sys
import dnn_misc
import os
import argparse

class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d, _, _ = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.X.shape[1], self.X.shape[2], self.X.shape[3]))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY


class CNN():
    def __init__(self, random_seed=2, learning_rate=0.01, alpha=0.0, lam=0.0, dropout_rate=0.5, num_epoch=30, minibatch_size=5):
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.lam = lam
        self.dropout_rate = dropout_rate
        self.num_epoch= num_epoch
        self.minibatch_size = minibatch_size

    def train(self, Xtrain, Ytrain, Xval, Yval):

        ### set the random seed ###
        np.random.seed(self.random_seed)

        ### data processing ###
        N_train, d, _, _ = Xtrain.shape
        N_val, _, _, _ = Xval.shape

        trainSet = DataSplit(Xtrain, Ytrain)
        valSet = DataSplit(Xval, Yval)

        ### building/defining CNN ###
        """
        The network structure is input --> convolution --> relu --> max pooling --> flatten --> dropout --> linear --> softmax_cross_entropy loss
        the hidden_layer size (num_L1) is 4900
        the output_layer size (num_L2) is 10
        """
        model = dict()
        num_L1 = 4900
        num_L2 = 10

        # optimization setting: _alpha for momentum, _lambda for weight decay
        _learning_rate = self.learning_rate
        _step = 10
        _alpha = self.alpha
        _lambda = self.lam
        _dropout_rate = self.dropout_rate

        # create objects (modules) from the module classes
        model['C1'] = dnn_misc.conv_layer(num_input = d, num_output = 25, filter_len = 5, stride = 1)
        model['nonlinear1'] = dnn_misc.relu()
        model['M1'] = dnn_misc.max_pool(max_len = 2, stride = 2)
        model['F1'] = dnn_misc.flatten_layer()
        model['drop1'] = dnn_misc.dropout(r = _dropout_rate)
        model['L1'] = dnn_misc.linear_layer(input_D = num_L1, output_D = num_L2)
        model['loss'] = dnn_misc.softmax_cross_entropy()

        # create variables for momentum
        if _alpha > 0.0:
            momentum = dnn_misc.add_momentum(model)
        else:
            momentum = None

        ### run training and validation ###
        for t in range(self.num_epoch):
            print('At epoch ' + str(t + 1))
            if (t % _step == 0) and (t != 0):
                _learning_rate = _learning_rate * 0.1

            idx_order = np.random.permutation(N_train)
            train_acc = 0.0
            train_loss = 0.0
            train_count = 0

            val_acc = 0.0
            val_count = 0

            for i in range(int(np.floor(N_train / self.minibatch_size))):

                # get a mini-batch of data
                x, y = trainSet.get_example(idx_order[i * self.minibatch_size : (i + 1) * self.minibatch_size])

                ### forward ###
                c1 = model['C1'].forward(x)
                h1 = model['nonlinear1'].forward(c1)
                m1 = model['M1'].forward(h1)
                f1 = model['F1'].forward(m1)
                d1 = model['drop1'].forward(f1, is_train = True)
                a1 = model['L1'].forward(d1)
                loss = model['loss'].forward(a1, y)

                ### backward ###
                grad_a1 = model['loss'].backward(a1, y)
                grad_d1 = model['L1'].backward(d1, grad_a1)
                grad_f1 = model['drop1'].backward(f1, grad_d1)
                grad_m1 = model['F1'].backward(m1, grad_f1)
                grad_h1 = model['M1'].backward(h1, grad_m1)
                grad_c1 = model['nonlinear1'].backward(c1, grad_h1)
                grad_x = model['C1'].backward(x, grad_c1)

                ### gradient_update ###
                for module_name, module in model.items():

                    # check if a module has learnable parameters
                    if hasattr(module, 'params'):
                        for key, _ in module.params.items():
                            g = module.gradient[key] + _lambda * module.params[key]

                            if _alpha > 0.0:
                                momentum[module_name + '_' + key] = _alpha * momentum[module_name + '_' + key] - _learning_rate * g
                                module.params[key] += momentum[module_name + '_' + key]

                            else:
                                module.params[key] -= _learning_rate * g

            ### Computing training accuracy and obj ###
            for i in range(int(np.floor(N_train / self.minibatch_size))):
                x, y = trainSet.get_example(np.arange(i * self.minibatch_size, (i + 1) * self.minibatch_size))

                ### forward ###
                c1 = model['C1'].forward(x)
                h1 = model['nonlinear1'].forward(c1)
                m1 = model['M1'].forward(h1)
                f1 = model['F1'].forward(m1)
                d1 = model['drop1'].forward(f1, is_train=False)
                a1 = model['L1'].forward(d1)
                loss = model['loss'].forward(a1, y)
                train_loss += len(y) * loss
                train_acc += np.sum(self.predict_label(a1) == y)
                train_count += len(y)

            train_loss = train_loss / train_count
            train_acc = train_acc / train_count

            print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
            print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

            ### Computing validation accuracy ###
            for i in range(int(np.floor(N_val / self.minibatch_size))):
                x, y = valSet.get_example(np.arange(i * self.minibatch_size, (i + 1) * self.minibatch_size))

                ### forward ###
                c1 = model['C1'].forward(x)
                h1 = model['nonlinear1'].forward(c1)
                m1 = model['M1'].forward(h1)
                f1 = model['F1'].forward(m1)
                d1 = model['drop1'].forward(f1, is_train=False)
                a1 = model['L1'].forward(d1)
                val_acc += np.sum(self.predict_label(a1) == y)
                val_count += len(y)

            val_acc = val_acc / val_count

            print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

        return model

    def predict(self, x, model):

        c1 = model['C1'].forward(x)
        h1 = model['nonlinear1'].forward(c1)
        m1 = model['M1'].forward(h1)
        f1 = model['F1'].forward(m1)
        d1 = model['drop1'].forward(f1, is_train = False)
        a1 = model['L1'].forward(d1)
        return self.predict_label(a1)

    def predict_label(self, f):
        # This is a function to determine the predicted label given scores
        if f.shape[1] == 1:
            return (f > 0).astype(float)
        else:
            return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))