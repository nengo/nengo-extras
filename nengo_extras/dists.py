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
