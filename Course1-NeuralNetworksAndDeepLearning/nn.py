#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Farmer Li (jxli@mobvoi.com)
# @Date:   2018-12-11

import numpy as np


def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    return A


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_backward(dA, cache):
    Z = cache
    s = sigmoid(Z)

    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


class DNN:
    def __init__(self, layer_dims, learning_rate=0.001):
        self.params = self.__init_params(layer_dims)
        self.learning_rate = learning_rate

    @staticmethod
    def __init_params(layer_dims):
        """Initialize the W and b of every layer.

        Arguments:
            layer_dims: python array (list) containing the dimensions of each layer in our network.

        Returns:
            params: python dictionary containing NN parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1]).
                    bl -- bias vector of shape (layer_dims[l], 1).
        """
        np.random.seed(1)

        params = {}
        for l in range(1, len(layer_dims)):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
            params['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1))

            assert (params['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (params['b' + str(l)].shape == (layer_dims[l], 1))

        return params

    @staticmethod
    def __linear_forward(A_prev, W, b):
        """Implement the linear part of a layer's forward propagation.

        Arguments:
            A_prev: Activations from previous layer(or input data).
            W: Weights matrix.
            b: Bias vector.

        Returns:
            Z: The input of activation function, also called pre-activation parameter.
            cache: A tuple containing "A_prev" and "W", stored for computing the backward pass.
        """

        Z = W.dot(A_prev) + b
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W)

        return Z, cache

    @staticmethod
    def __linear_activation_forward(A_prev, W, b, activation):
        """Implement the forward propagation of the complete neuron.

        Arguments:
            A_prev: Activations from previous layer(or input data).
            W: Weights matrix.
            b: Bias vector.
            activation: Activation function name, only 'sigmoid' and 'relu' supported.
                        If not both, the activation output will be equal to Z.

        Returns:
            A: The output of activation function, also called post-activation value.
            cache: A tuple containing "linear_cache" and "activation_cache", stored for computing the backward pass.

        """

        Z, linear_cache = DNN.__linear_forward(A_prev, W, b)

        if activation == 'sigmoid':
            A = sigmoid(Z)
        elif activation == 'relu':
            A = relu(Z)
        else:
            print('!!Warning: None activation. A == Z.')
            A = Z

        activation_cache = Z
        cache = (linear_cache, activation_cache)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        return A, cache

    @staticmethod
    def __forward(params, X):
        """Implement forward propagation of [LINEAR->RELU] * (L-1) -> [LINEAR->SIGMOID] neural network.

        Arguments:
            X: Data, numpy array

        Returns:
            AL: Last activation output.
            caches: List of caches containing:
                    Every cache of linear->relu layer (there are L-1 of them, note indexed from 0 to L-2)
                    The cache of linear->sigmoid layer (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(params) // 2

        # Implement [LINEAR->RELU] * (L-1), add 'cache' to 'caches' list.
        for l in range(1, L):
            A_prev = A
            A, cache = DNN.__linear_activation_forward(A_prev, params['W' + str(l)], params['b' + str(l)], 'relu')
            caches.append(cache)

        # Implement LINEAR->SIGMOID, add 'cache' to 'caches' list.
        AL, cache = DNN.__linear_activation_forward(A, params['W' + str(L)], params['b' + str(L)], 'sigmoid')
        caches.append(cache)

        assert (AL.shape == (params['W' + str(L)].shape[0], X.shape[1]))

        return AL, caches

    @staticmethod
    def __compute_cost(AL, Y):
        """Implement of Cross-entropy cost function.

        Arguments:
            AL: Probability corresponding to labels in Y.
            Y: True 'label' vector.

        Returns:
            cost: Cross-entropy cost.
        """

        m = Y.shape[1]

        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

        cost = np.squeeze(cost)

        assert (cost.shape == ())

        return cost

    @staticmethod
    def __linear_backward(dZ, cache):
        """Implement the linear portion of backward propagation for a single layer(layer l).

        Arguments:
            dZ: Gradient of the cost with respect to the linear output (of current layer l).
            cache: Tuple of values(A_prev, W) coming from the forward propagation in the current layer.

        Returns:
            dA_prev: Gradient of the cost whit respect to the activation (of the previous layer l-1).
            dW: Gradient of the cost with respect to W (of current layer l).
            db: Gradient of the cost with respect to b (of current layer l).
        """
        A_prev, W = cache
        m = A_prev.shape[1]

        dW = dZ.dot(A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = W.T.dot(dZ)

        assert (dW.shape == W.shape)
        assert (db.shape == (W.shape[0], 1))
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    @staticmethod
    def __linear_activation_backward(dA, cache, activation):
        """Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
            dA: Post-activation gradient for current layer l.
            cache: Tuple of values (linear_cache, activation_cache)
            activation: Activation function name, only 'sigmoid' and 'relu' supported.
                        If not both, the post-activation gradient will be equal to dZ.

        Returns:
            dA_prev: Post-activation gradient for current layer l-1.
        """
        linear_cache, activation_cache = cache

        if activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == 'relu':
            dZ = relu_backward(dA, activation_cache)
        else:
            dZ = dA

        dA_prev, dW, db = DNN.__linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    @staticmethod
    def __backward(AL, Y, caches):
        """Implement the backward for the [LINEAR -> RELU] * (L-1) -> LINEAR->SIGMOID network.

        Arguments:
            AL: Probability corresponding to labels in Y.
            Y: True 'label' vector.
            caches: List of caches containing:
                    Every cache of linear->relu layer (there are L-1 of them, note indexed from 0 to L-2)
                    The cache of linear->sigmoid layer (there is one, indexed L-1)

        Returns:
            grads: A dictionary with the gradient.
                   grads['dA' + str(l)] denote gradient for post-activation value of layer l-1
                   grads['dW' + str(l)] denote gradient for weight matrix of layer l
                   grads['db' + str(l)] denote gradient for bias of layer l
        """
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        # Initialize the backward propagation.
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer(SIGMOID->LINEAR) gradient.
        current_cache = caches[-1]
        dA_prev, dW, db = DNN.__linear_activation_backward(dAL, current_cache, 'sigmoid')
        grads['dA' + str(L)] = dA_prev
        grads['dW' + str(L)] = dW
        grads['db' + str(L)] = db

        for l in reversed(range(L - 1)):
            current_cache = caches[l]  # Take care of this index.
            dA_prev, dW, db = DNN.__linear_activation_backward(grads['dA' + str(l + 2)], current_cache, 'relu')
            grads['dA' + str(l + 1)] = dA_prev  # '_prev' represents the previous layer relative to layer l
            grads['dW' + str(l + 1)] = dW
            grads['db' + str(l + 1)] = db

        return grads

    @staticmethod
    def __update_parameters(params, grads, learning_rate):
        """Update parameter using gradients with specified learning rate.

        Arguments:
            params: Network weight matrix and bias vector parameters.
            grads: Gradients of each layer of neural network.
            learning_rate: Learning rate.

        Returns:
            params: Updated parameters.
        """
        L = len(params) // 2
        for l in range(1, L + 1):
            index_str = str(l)
            params['W' + index_str] -= learning_rate * grads['dW' + index_str]
            params['b' + index_str] -= learning_rate * grads['db' + index_str]

        return params

    def train(self, X, Y):
        """Implement of single training cycle. This function may require multiple calls to get better results.

        Arguments:
            X: Training dataset.
            Y: Labels vector corresponding to the training data.

        Returns:
            cost: Current Cross-entropy cost.
        """
        # Forward propagation
        AL, caches = self.__forward(self.params, X)

        # Compute cost.
        cost = self.__compute_cost(AL, Y)

        # Backward propagation
        grads = self.__backward(AL, Y, caches)

        # Update parameters
        self.params = self.__update_parameters(self.params, grads, self.learning_rate)

        return cost

    def model(self, X, Y, num_iterations=3000, print_cost=False):
        """Training neural network with specified iterations.

        Arguments:
            X: Training dataset.
            Y: Labels vector corresponding to the training data.
            num_iterations: Gradient descent algorithm iterations.
            print_cost: If True, print the cost value during training (every 100 steps).

        Returns:
            costs: Cross-entropy cost for every 100 steps.
        """
        costs = []

        for i in range(num_iterations):
            cost = self.train(X, Y)
            if i % 100 == 0:
                costs.append(cost)
                if print_cost:
                    print("Cost after iteration %i: %f" % (i, cost))
        return costs

    def predict(self, data):
        """Implement of predict, convert probability vector to label(0 or 1) vector.

        Arguments:
            data: Data to be predict.

        Returns:
            predicted: Predicted labels.
        """
        # Forward propagation
        probability, _ = self.__forward(self.params, data)

        predicted = np.asarray([probability >= 0.5], int)

        return predicted
