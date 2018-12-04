#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Farmer Li (jxli@mobvoi.com)
# @Date:   2018-12-04 12:03

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1)  # set a seed so that the results are consistent

X, Y = load_planar_dataset()
# Check the data shape.
print('X shape is: ', X.shape)
print('Y shape is: ', Y.shape)


# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)
plt.show()
