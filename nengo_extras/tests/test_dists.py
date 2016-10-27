import numpy as np

from nengo_extras.dists import MultivariateGaussian


def test_multivariate_gaussian(plt, rng):
    n = 200000

    dist = MultivariateGaussian((-1, 1), (0.25, 4))
    pts = dist.sample(n, rng=rng)

    plt.hist2d(pts[:, 0], pts[:, 1], bins=np.arange(-5, 5, 0.2))
