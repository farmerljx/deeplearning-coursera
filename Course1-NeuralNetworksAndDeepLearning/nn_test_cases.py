#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Farmer Li (jxli@mobvoi.com)
# @Date:   2018-12-12

import numpy as np
import h5py
import matplotlib.pyplot as plt

from nn import DNN
from week4.testCases import *


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def test_dnn():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    learning_rate = 0.0075
    dnn = DNN(layers_dims, learning_rate=learning_rate)

    costs = dnn.model(train_x, train_y, num_iterations=2500, print_cost=True)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predicted_train = dnn.predict(train_x)
    accuracy_train = np.sum(predicted_train == train_y) / train_y.shape[1]
    print('\nTrain accuracy: ', accuracy_train)

    predicted_test = dnn.predict(test_x)
    accuracy_test = np.sum(predicted_test == test_y) / test_y.shape[1]
    print('Test  accuracy: ', accuracy_test)


def main():
    print('Running DNN test case...')
    test_dnn()


if __name__ == '__main__':
    main()
