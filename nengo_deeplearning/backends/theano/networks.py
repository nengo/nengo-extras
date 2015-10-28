"""Neural networks implemented with Theano.

Culled from https://github.com/IndicoDataSolutions/Passage (MIT license).
"""

import sys
from time import time

import theano
import theano.tensor as T
import numpy as np

from . import costs, iterators, optimizers
from .utils import case_insensitive_import, flatten, list_index, one_hot, save


def standardize_targets(Y, cost):
    Y = np.asarray(Y)
    ndim = len(Y.shape)
    if ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[1] == 1 and cost.__name__ == 'CategoricalCrossEntropy':
        Y = one_hot(Y, negative_class=0.)
    if Y.shape[1] == 1 and 'Hinge' in cost.__name__:
        if len(np.unique(Y)) > 2:
            Y = one_hot(Y, negative_class=-1.)
        else:
            Y[Y == 0] -= 1
    return Y


class LenFilter(object):
    def __init__(self, max_len=1000, min_max_len=100, percentile=99):
        self.max_len = max_len
        self.percentile = percentile
        self.min_max_len = min_max_len

    def filter(self, *data):
        lens = [len(seq) for seq in data[0]]
        if self.percentile > 0:
            max_len = np.percentile(lens, self.percentile)
            max_len = np.clip(max_len, self.min_max_len, self.max_len)
        else:
            max_len = self.max_len
        valid_idxs = [i for i, l in enumerate(lens) if l <= max_len]
        if len(data) == 1:
            return list_index(data[0], valid_idxs)
        else:
            return tuple([list_index(d, valid_idxs) for d in data])


class RNN(object):
    def __init__(self, layers, cost, verbose=2,
                 optimizer='Adam', Y=T.matrix(), iterator='SortedPadded'):
        self.settings = {'layers': layers,
                         'cost': cost,
                         'verbose': verbose,
                         'optimizer': optimizer,
                         'Y': Y,
                         'iterator': iterator}
        self.layers = layers

        if isinstance(cost, basestring):
            self.cost = case_insensitive_import(costs, cost)
        else:
            self.cost = cost

        if isinstance(optimizer, basestring):
            self.optimizer = case_insensitive_import(optimizers, optimizer)()
        else:
            self.optimizer = optimizer

        if isinstance(iterator, basestring):
            self.iterator = case_insensitive_import(iterators, iterator)()
        else:
            self.iterator = iterator

        self.verbose = verbose
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in layers])

        self.X = self.layers[0].input
        self.y_tr = self.layers[-1].output(dropout_active=True)
        self.y_te = self.layers[-1].output(dropout_active=False)
        self.Y = Y

        cost = self.cost(self.Y, self.y_tr)
        self.updates = self.optimizer.get_updates(self.params, cost)

        self._train = theano.function([self.X, self.Y], cost,
                                      updates=self.updates)
        self._cost = theano.function([self.X, self.Y], cost)
        self._predict = theano.function([self.X], self.y_te)

    def fit(self, trX, trY, batch_size=64, n_epochs=1,
            len_filter=LenFilter(), snapshot_freq=1, path=None):
        """Train model on given training examples.

        Returns the list of costs after each minibatch is processed.

        Parameters
        ----------
        trX : list
            Inputs
        trY : list
            Outputs
        batch_size : int, optional
            Number of examples in a minibatch. Default: 64
        n_epochs : int, optional
            Number of epochs to train for. Default: 1
        len_filter : object, optional
            Object to filter training example by length. Default: LenFilter()
        snapshot_freq : int, optional
            Number of epochs between saving model snapshots. Default: 1
        path : str, optional
            Prefix of path where model snapshots are saved.
            If `None`, no snapshots are saved. Default: None

        Returns
        -------
        costs : list
          Costs of model after processing each minibatch
        """
        if len_filter is not None:
            trX, trY = len_filter.filter(trX, trY)
        trY = standardize_targets(trY, cost=self.cost)

        n = 0.
        t = time()
        costs = []
        for e in range(n_epochs):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY) - n % len(trY)
                    time_left = n_left / n_per_sec
                    sys.stdout.write("\rEpoch %d\tSeen %d samples\tAvg cost "
                                     "%0.4f\tTime left %d seconds" % (
                                         e, n, np.mean(epoch_costs[-250:]),
                                         time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)

            status = ("Epoch %d\tSeen %d samples\tAvg cost %0.4f\tTime elapsed"
                      " %d seconds" % (
                          e, n, np.mean(epoch_costs[-250:]), time() - t))
            if self.verbose >= 2:
                sys.stdout.write("\r"+status)
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
            if path and e % snapshot_freq == 0:
                save(self, "{0}.{1}".format(path, e))
        return costs

    def predict(self, X):
        if isinstance(self.iterator, (iterators.Padded, iterators.Linear)):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]
