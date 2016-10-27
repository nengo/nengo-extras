import numpy as np

from nengo.dists import Distribution
from nengo.params import NdarrayParam


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
