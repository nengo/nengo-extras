import time

import numpy as np

import nengo.solvers
import nengo.utils.numpy as npext
from nengo.params import BoolParam, IntParam, NumberParam

from nengo_extras.convnet import softmax


def cho_solve(A, y, overwrite=False):
    """Helper to solve ``A x = y`` for ``x`` using Cholesky decomposition.
    """
    try:
        import scipy.linalg
        factor = scipy.linalg.cho_factor(A, overwrite_a=overwrite)
        x = scipy.linalg.cho_solve(factor, y)
    except ImportError:
        L = np.linalg.cholesky(A)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, y))

    return x


class LstsqClassifier(nengo.solvers.Solver):
    """Minimize the weighted squared loss for one-hot classification.

    Uses weighted least squares to solve for better classification weights.

    Parameters
    ----------
    weights : bool, optional (Default: False)
        If False, solve for decoders. If True, solve for weights.
    reg : float, optional (Default: 0.01)
        Amount of L2 regularization, as a fraction of the neuron activity.
    weight_power : float, optional (Default: 1)
        Exponent for the weights.

    Attributes
    ----------
    precompute_ai : bool (Default: True)
        Whether to precompute the subcomponents of the Gram matrix. Much faster
        computation at the expense of slightly more memory.
    """

    reg = NumberParam('reg', low=0)
    weight_power = NumberParam('weight_power', low=0)
    precompute_ai = BoolParam('precompute_ai')

    def __init__(self, weights=False, reg=0.01, weight_power=1):
        super(LstsqClassifier, self).__init__(weights=weights)
        self.reg = reg
        self.weight_power = weight_power
        self.precompute_ai = True

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()

        m, n = A.shape
        _, d = Y.shape
        sigma = self.reg * A.max()
        precompute_ai = self.precompute_ai

        Y = Y > 0.5  # ensure Y is binary

        def getAAi(i, y, cache={}):
            if i in cache:
                return cache[i]

            Ai = A[y]
            AAi = np.dot(Ai.T, Ai)
            if precompute_ai:
                cache[i] = AAi
            return AAi

        if not precompute_ai:
            AA = np.dot(A.T, A)
        else:
            AA = np.zeros((n, n))
            for i in range(d):
                AA += getAAi(i, Y[:, i])

        X = np.zeros((n, d))
        for i in range(d):
            y = Y[:, i]

            # weight for classification
            p = y.mean()
            q = self.weight_power
            wr = p*(1-p)**q + (1-p)*p**q
            w0 = p**q / wr
            w1 = (1-p)**q / wr
            dw = w1 - w0
            w = w0 + dw*y

            # form Gram matrix G = A.T W A + m * sigma**2
            G = w0*AA + dw*getAAi(i, y)
            np.fill_diagonal(G, G.diagonal() + m * sigma**2)
            b = np.dot(A.T, w * y)

            X[:, i] = cho_solve(G, b, overwrite=True)

        tend = time.time()
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart}


class HingeClassifier(nengo.solvers.Solver):
    """Minimize the hinge loss for a one-hot classification output.

    Parameters
    ----------
    weights : bool, optional (Default: False)
        If False, solve for decoders. If True, solve for weights.
    reg : float, optional (Default: 0.01)
        Amount of L2 regularization, as a fraction of the neuron activity.
    maxfun : int, optional (Default: 1000)
        Maximum number of gradient function evaluations.
    verbose : int, optional (Default: 0)
        Verbosity of the solver (see ``scipy.optimize.fmin_l_bfgs_b:iprint``).
    """

    reg = NumberParam('reg', low=0)
    maxfun = IntParam('maxfun', low=1)
    verbose = IntParam('verbose')

    def __init__(self, weights=False, reg=0.01, maxfun=1000, verbose=0):
        import scipy.optimize
        self.scipy_opt = scipy.optimize

        super(HingeClassifier, self).__init__(weights=weights)
        self.reg = reg
        self.maxfun = maxfun
        self.verbose = verbose

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()

        assert A.shape[0] == Y.shape[0]
        m, n = A.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = self.reg * A.max()
        lamb = m * sigma**2

        # --- initialization
        X0 = rng.uniform(-1./n, 1./n, size=Xshape)
        # X0, _ = nengo.solvers.LstsqL2(reg=self.reg)(A, Y, rng=rng, E=E)

        # --- solve with L-BFGS
        yi = Y.argmax(axis=1)

        def f_df(x):
            X = x.reshape(Xshape)
            Z = np.dot(A, X)

            # Crammer and Singer (2001) version
            zy = Z[np.arange(m), yi]
            Z[np.arange(m), yi] = -np.inf
            ti = Z.argmax(axis=1)
            zt = Z[np.arange(m), ti]
            margins = zy - zt

            E = np.zeros(Z.shape)
            margin1 = margins < 1
            E[margin1, yi[margin1]] = -1
            E[margin1, ti[margin1]] = 1

            cost = np.maximum(0, 1 - margins).sum()
            grad = np.dot(A.T, E)
            if lamb > 0:
                cost += 0.5 * lamb * (X**2).sum()
                grad += lamb * X
            return cost, grad.ravel()

        w0 = X0.ravel()
        w, mincost, info = self.scipy_opt.fmin_l_bfgs_b(
            f_df, w0, maxfun=self.maxfun, iprint=self.verbose)

        t = time.time() - tstart

        X = w.reshape(Xshape)
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': t,
            'iterations': info['funcalls'],
        }


class SoftmaxClassifier(nengo.solvers.Solver):
    """Solver to minimize the softmax loss for a one-hot classification output.

    The softmax loss is also known as the negative log likelihood.
    It is commonly used in artificial neural network classifiers.

    Parameters
    ----------
    weights : bool, optional (Default: False)
        If False, solve for decoders. If True, solve for weights.
    reg : float, optional (Default: 2e-3)
        Amount of L2 regularization, as a fraction of the neuron activity.
    maxfun : int, optional (Default: 1000)
        Maximum number of gradient function evaluations.
    verbose : int, optional (Default: 0)
        Verbosity of the solver (see ``scipy.optimize.fmin_l_bfgs_b:iprint``).
    """

    reg = NumberParam('reg', low=0)
    maxfun = IntParam('maxfun', low=1)
    verbose = IntParam('verbose')

    def __init__(self, weights=False, reg=2e-3, maxfun=1000, verbose=0):
        import scipy.optimize
        self.scipy_opt = scipy.optimize

        super(SoftmaxClassifier, self).__init__(weights=weights)
        self.reg = reg
        self.maxfun = maxfun
        self.verbose = verbose

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()

        assert A.shape[0] == Y.shape[0]
        m, n = A.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = self.reg * A.max()
        lamb = m * sigma**2

        # --- initialization
        X0 = np.zeros(Xshape)
        # X0, _ = nengo.solvers.LstsqL2(reg=self.reg)(A, Y, rng=rng, E=E)

        # --- solve with L-BFGS
        yi = Y.argmax(axis=1)
        mi = np.arange(m)

        def f_df(x):
            X = x.reshape(Xshape)
            Yest = softmax(np.dot(A, X), axis=1)
            cost = -np.log(np.maximum(Yest[mi, yi], 1e-16)).sum()
            E = Yest - Y
            grad = np.dot(A.T, E)
            if lamb > 0:
                cost += 0.5 * lamb * (X**2).sum()
                grad += lamb * X
            return cost, grad.ravel()

        x0 = X0.ravel()
        x, mincost, info = self.scipy_opt.fmin_l_bfgs_b(
            f_df, x0, maxfun=self.maxfun, iprint=self.verbose)

        t = time.time() - tstart

        X = x.reshape(Xshape)
        return self.mul_encoders(X, E), {
            'rmses': npext.rms(softmax(np.dot(A, X), axis=1) - Y, axis=1),
            'time': t,
            'iterations': info['funcalls'],
        }


nengo.cache.Fingerprint.whitelist(LstsqClassifier)
nengo.cache.Fingerprint.whitelist(HingeClassifier)
nengo.cache.Fingerprint.whitelist(SoftmaxClassifier)
