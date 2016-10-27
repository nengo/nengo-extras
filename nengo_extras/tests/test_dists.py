import numpy as np

from nengo.dists import Gaussian, Uniform
from nengo_extras.dists import Mixture, MultivariateGaussian


def test_multivariate_gaussian(plt, rng):
    n = 200000

    dist = MultivariateGaussian((-1, 1), (0.25, 4))
    pts = dist.sample(n, rng=rng)

    plt.hist2d(pts[:, 0], pts[:, 1], bins=np.arange(-5, 5, 0.2))


def test_mixture_1d(plt, rng):
    n = 100000
    bins = np.arange(-5, 5, 0.1)

    dists = [Uniform(-0.5, 0.5), Gaussian(-1, 1), Gaussian(2, 1)]
    mdists = [Mixture(dists),
              Mixture(dists, p=[0.7, 0.15, 0.15])]

    for k, dist in enumerate(mdists):
        plt.subplot(len(mdists), 1, k+1)
        pts = dist.sample(n, rng=rng)
        plt.hist(pts, bins=bins)


def test_mixture_2d(plt, rng):
    n = 200000

    dists = [MultivariateGaussian((-2, -1), (0.25, 4)),
             MultivariateGaussian((1, 1), (1, 1))]
    mdists = [Mixture(dists),
              Mixture(dists, p=[0.7, 0.3])]

    for k, dist in enumerate(mdists):
        plt.subplot(len(mdists), 1, k+1)
        pts = dist.sample(n, d=2, rng=rng)
        plt.hist2d(pts[:, 0], pts[:, 1], bins=np.arange(-5, 5, 0.2))
