from __future__ import absolute_import

import numpy as np

from nengo.dists import Distribution, DistributionParam, Uniform
from nengo.params import NdarrayParam, NumberParam, TupleParam


def gaussian_icdf(mean, std):
    import scipy.stats as sps

    def icdf(p):
        return sps.norm.ppf(p, scale=std, loc=mean)

    return icdf


def loggaussian_icdf(log_mean, log_std, base=np.e):
    import scipy.stats as sps

    mean = base ** log_mean
    log_std2 = log_std * np.log(base)

    def icdf(p):
        return sps.lognorm.ppf(p, log_std2, scale=mean)

    return icdf


def uniform_icdf(low, high):
    def icdf(p):
        return p * (high - low) + low

    return icdf


class Concatenate(Distribution):
    """Concatenate distributions to form an independent multivariate"""

    distributions = TupleParam("distributions", readonly=True)
    d = NumberParam("d", low=1, readonly=True)

    def __init__(self, distributions):
        super(Concatenate, self).__init__()
        self.distributions = distributions

        # --- determine dimensionality
        rng = np.random.RandomState(0)
        s = np.column_stack([d.sample(1, rng=rng) for d in self.distributions])
        self.d = s.shape[1]

    def sample(self, n, d=None, rng=np.random):
        assert d is None or d == self.d
        return np.column_stack([dist.sample(n, rng=rng) for dist in self.distributions])


class MultivariateCopula(Distribution):
    """Generalized multivariate distribution.

    Uses the copula method to sample from a general multivariate distribution,
    given marginal distributions and copula covariances [1]_.

    Parameters
    ----------
    marginal_icdfs : iterable
        List of functions, each one being the inverse CDF of the marginal
        distribution across that dimension.
    rho : array_like (optional)
        Array of copula covariances [1]_ between parameters. Defaults to
        the identity matrix (independent parameters).

    See also
    --------
    gaussian_icdf, loggaussian_icdf, uniform_icdf

    References
    ----------
    .. [1] Copula (probability theory). Wikipedia.
       https://en.wikipedia.org/wiki/Copula_(probability_theory%29
    """

    marginal_icdfs = TupleParam("marginal_icdfs", readonly=True)
    rho = NdarrayParam("rho", shape=("*", "*"), optional=True, readonly=True)

    def __init__(self, marginal_icdfs, rho=None):
        import scipy.stats  # we need this for sampling

        assert scipy.stats

        super(MultivariateCopula, self).__init__()
        self.marginal_icdfs = marginal_icdfs
        self.rho = rho

        d = len(self.marginal_icdfs)
        if not all(callable(f) for f in self.marginal_icdfs):
            raise ValueError("`marginal_icdfs` must be a list of callables")
        if self.rho is not None:
            if self.rho.shape != (d, d):
                raise ValueError("`rho` must be a %d x %d array" % (d, d))
            if not np.array_equal(self.rho, self.rho.T):
                raise ValueError("`rho` must be a symmetrical positive-definite array")

    def sample(self, n, d=None, rng=np.random):
        import scipy.stats as sps

        assert d is None or d == len(self.marginal_icdfs)
        d = len(self.marginal_icdfs)

        # normalize rho
        rho = np.eye(d) if self.rho is None else self.rho
        stds = np.sqrt(np.diag(rho))
        rho = rho / np.outer(stds, stds)

        # sample from copula
        x = sps.norm.cdf(sps.multivariate_normal.rvs(cov=rho, size=n))

        # apply marginal inverse CDFs
        for i in range(d):
            x[:, i] = self.marginal_icdfs[i](x[:, i])

        return x


class MultivariateGaussian(Distribution):
    mean = NdarrayParam("mean", shape="d")
    cov = NdarrayParam("cov", shape=("d", "d"))

    def __init__(self, mean, cov):
        super(MultivariateGaussian, self).__init__()

        self.d = len(mean)
        self.mean = mean
        cov = np.asarray(cov)
        self.cov = (
            cov * np.eye(self.d)
            if cov.size == 1
            else np.diag(cov)
            if cov.ndim == 1
            else cov
        )

    def sample(self, n, d=None, rng=np.random):
        assert d is None or d == self.d
        return rng.multivariate_normal(self.mean, self.cov, size=n)


class Mixture(Distribution):
    distributions = TupleParam("distributions")
    p = NdarrayParam("p", shape="*", optional=True)

    def __init__(self, distributions, p=None):
        super(Mixture, self).__init__()

        self.distributions = distributions
        if not all(isinstance(d, Distribution) for d in self.distributions):
            raise ValueError("All elements in `distributions` must be Distributions")

        if p is not None:
            p = np.array(p)
            if p.ndim != 1 or p.size != len(self.distributions):
                raise ValueError(
                    "`p` must be a vector with one element per distribution"
                )
            if (p < 0).any():
                raise ValueError("`p` must be all non-negative")
            p /= p.sum()
        self.p = p

    def sample(self, n, d=None, rng=np.random):
        dd = 1 if d is None else d
        samples = np.zeros((n, dd))

        ndims = len(self.distributions)
        i = (
            rng.randint(ndims, size=n)
            if self.p is None
            else rng.choice(ndims, p=self.p, size=n)
        )
        c = np.bincount(i, minlength=ndims)

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

    values = NdarrayParam("values", shape=("*", "*"))

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
            values = np.tile(
                self.values, (int(np.ceil(float(n) / nv)), int(np.ceil(float(d) / dv)))
            )
        else:
            values = self.values

        values = values[:n, :d]
        return values[:, 0] if out1 else values


class AreaIntercepts(Distribution):
    """ In 1 dimension, a uniform distribution of intercepts will lead to a uniform
    distribution of the % of state space a neuron is active in. As the dimensionality
    of the represented state space increases, a uniform distribution of intercepts
    leads to neurons that are active all the time or none of the time. The
    AreaIntercepts distribution transforms intercepts such that neurons in state spaces
    > 1D are active for the same % of state space as they would be in 1 dimensional
    state space, by reworking the volume of a hyperspehere cap equation.
    """

    dimensions = NumberParam("dimensions")
    base = DistributionParam("base")

    def __init__(self, dimensions, base=Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return "AreaIntercepts(dimensions=%r, base=%r)" % (self.dimensions, self.base)

    def transform(self, x):
        """ Transform the intercepts to account for the volume of the hypersphere cap
        where neurons will be active such that the % of state space they are active in
        is consistent with 1D state space.
        """
        import scipy.special  # noqa

        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(
            1 - scipy.special.betaincinv((self.dimensions + 1) / 2.0, 0.5, x + 1)
        )

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for ii, ss in enumerate(s):
            s[ii] = self.transform(ss)
        return s


class Triangular(Distribution):
    """ Generate samples using a triangular distribution between an upper (right) and
    lower (left) bound around a mode.
    """

    left = NumberParam("left")
    right = NumberParam("right")
    mode = NumberParam("mode")

    def __init__(self, left, mode, right):
        super(Triangular, self).__init__()
        self.left = left
        self.right = right
        self.mode = mode

    def __repr__(self):
        return "Triangular(left=%r, mode=%r, right=%r)" % (
            self.left,
            self.mode,
            self.right,
        )

    def sample(self, n, d=None, rng=np.random):
        return -1 * rng.triangular(
            self.left, self.mode, self.right, size=n if d is None else (n, d)
        )
