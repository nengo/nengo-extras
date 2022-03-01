import time

import numpy as np
from scipy.optimize import nnls

from nengo.params import NumberParam
from nengo.solvers import Solver
from nengo.utils.least_squares_solvers import format_system, rmses


class Dales(Solver):
    """Solves for weights subject to Dale's principle."""

    p_inh = NumberParam('p_inh', low=0, high=1)

    def __init__(self, p_inh=0.2):
        super(Dales, self).__init__(weights=True)
        self.p_inh = p_inh

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        d = Y.shape[1]

        Y = self.mul_encoders(Y, E)
        i = int(self.p_inh * n)
        A[:, :i] *= (-1)

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for j in range(d):
            X[:, j], residuals[i] = nnls(A, Y[:, j])
        X[:i, :] *= (-1)

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y),
                'residuals': residuals,
                'time': t,
                'i': i}

        return X, info


class DalesL2(Dales):
    """Solves for weights subject to Dale's principle with regularisation."""

    def __call__(self, A, Y, rng=None, E=None, sigma=0.):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        d = Y.shape[1]

        Y = self.mul_encoders(Y, E)
        i = int(self.p_inh * n)
        A[:, :i] *= (-1)

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma ** 2)
        GY = np.dot(A.T, Y.clip(0, None))

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for j in range(d):
            X[:, j], residuals[i] = nnls(A, Y[:, j])
        X[:i, :] *= (-1)

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y),
                'residuals': residuals,
                'time': t,
                'i': i}

        return X, info
