from __future__ import absolute_import

import numpy as np

from nengo.dists import Distribution
from nengo.params import NdarrayParam, TupleParam


class MultivariateGaussian(Distribution):
    mean = NdarrayParam('mean', shape='d')
    cov = NdarrayParam('cov', shape=('d', 'd'))

    def __init__(self, mean, cov):
        super(MultivariateGaussian, self).__init__()

        self.d = len(mean)
        self.mean = mean
        cov = np.asarray(cov)
        self.cov = (cov*np.eye(self.d) if cov.size == 1 else
                    np.diag(cov) if cov.ndim == 1 else cov)

    def sample(self, n, d=None, rng=np.random):
        assert d is None or d == self.d
        return rng.multivariate_normal(self.mean, self.cov, size=n)


class Mixture(Distribution):
    distributions = TupleParam('distributions')
    p = NdarrayParam('p', shape='*', optional=True)

    def __init__(self, distributions, p=None):
        super(Mixture, self).__init__()

        self.distributions = distributions
        if not all(isinstance(d, Distribution) for d in self.distributions):
            raise ValueError(
                "All elements in `distributions` must be Distributions")

        if p is not None:
            p = np.array(p)
            if p.ndim != 1 or p.size != len(self.distributions):
                raise ValueError(
                    "`p` must be a vector with one element per distribution")
            if (p < 0).any():
                raise ValueError("`p` must be all non-negative")
            p /= p.sum()
        self.p = p

    def sample(self, n, d=None, rng=np.random):
        dd = 1 if d is None else d
        samples = np.zeros((n, dd))

        nd = len(self.distributions)
        i = (rng.randint(nd, size=n) if self.p is None else
             rng.choice(nd, p=self.p, size=n))
        c = np.bincount(i, minlength=nd)

        for k in c.nonzero()[0]:
            samples[i == k] = self.distributions[k].sample(c[k], d=dd, rng=rng)

        return samples[:, 0] if d is None else samples


class Tile(Distribution):
    """Choose values in order from an array

    This distribution is not random, but rather tiles an array to be a
    particular size. This is useful for example if you want to pass an array
    for a neuron parameter, but are not sure how many neurons there will be.

    Parameters
    ----------
    values : array_like
        The values to tile.
    """

    values = NdarrayParam('values', shape=('*', '*'))

    def __init__(self, values):
        super(Tile, self).__init__()

        values = np.asarray(values)
        self.values = values.reshape(-1, 1) if values.ndim < 2 else values

    def __repr__(self):
        return "Tile(values=%s)" % (self.values)

    def sample(self, n, d=None, rng=np.random):
        out1 = d is None
        d = 1 if d is None else d
        nv, dv = self.values.shape

        if n > nv or d > dv:
            values = np.tile(self.values, (int(np.ceil(float(n) / nv)),
                                           int(np.ceil(float(d) / dv))))
        else:
            values = self.values

        values = values[:n, :d]
        return values[:, 0] if out1 else values
