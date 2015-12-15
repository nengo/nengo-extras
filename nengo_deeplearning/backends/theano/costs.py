"""Cost functions for Theano networks.

Culled from https://github.com/IndicoDataSolutions/Passage (MIT license).
"""

import theano.tensor as T


def categorical_crossentropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()


def binary_crossentropy(y_true, y_pred):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()


def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()


cce = categorical_crossentropy
bce = binary_crossentropy
mse = mean_squared_error
mae = mean_absolute_error
