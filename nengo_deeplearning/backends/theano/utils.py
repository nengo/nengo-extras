"""Miscellaneous utilities for Theano networks.

Culled from https://github.com/IndicoDataSolutions/Passage (MIT license).
"""
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import theano


# --- Theano utils
def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name)


def downcast_float(X):
    return np.asarray(X, dtype=np.float32)


# --- General utils

def flatten(l):
    return [item for sublist in l for item in sublist]


def list_index(l, idxs):
    return [l[idx] for idx in idxs]


def one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        yield b


def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name.lower()])


def load(path):
    from . import layers
    from . import models
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    model_class = getattr(models, model['model'])
    model['config']['layers'] = [
        getattr(layers, layer['layer'])(**layer['config'])
        for layer in model['config']['layers']
    ]
    model = model_class(**model['config'])
    return model


def save(model, path):
    layer_configs = []
    for layer in model.layers:
        layer_config = layer.settings
        layer_name = layer.__class__.__name__
        weights = [p.get_value() for p in layer.params]
        layer_config['weights'] = weights
        layer_configs.append({'layer': layer_name, 'config': layer_config})
    model.settings['layers'] = layer_configs
    serializable_model = {
        'model': model.__class__.__name__,
        'config': model.settings
    }
    with open(path, 'wb') as fp:
        pickle.dump(serializable_model, fp)
