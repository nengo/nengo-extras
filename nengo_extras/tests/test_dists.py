import pytest
import numpy as np
import scipy.special

import nengo
from nengo.dists import Gaussian, Uniform, Distribution
from nengo_extras.dists import (
    Concatenate,
    gaussian_icdf,
    loggaussian_icdf,
    Mixture,
    MultivariateCopula,
    MultivariateGaussian,
    Tile,
    uniform_icdf,
    Triangular,
    AreaIntercepts,
)


def test_concatenate(plt, rng):
    n = 10000

    dist = Concatenate(
        [
            Uniform(-1, 1),
            Uniform(0, 1),
            MultivariateGaussian([0, 2], [2, 1]),
            Gaussian(3, 0.5),
        ]
    )
    pts = dist.sample(n, rng=rng)
    assert pts.shape == (n, 5)
    n, d = pts.shape

    for i in range(d):
        plt.subplot(d, 1, i + 1)
        plt.hist(pts[:, i], bins=np.linspace(-4, 4, 101))


def test_icdfs(plt, rng):
    p = rng.rand(100000)

    rows = 3
    plt.subplot(rows, 1, 1)
    plt.hist(gaussian_icdf(1, 0.5)(p), bins=51)

    plt.subplot(rows, 1, 2)
    h, b = np.histogram(loggaussian_icdf(-2, 0.5, base=10)(p), bins=np.logspace(-4, 0))
    plt.semilogx(0.5 * (b[:-1] + b[1:]), h)

    plt.subplot(rows, 1, 3)
    plt.hist(uniform_icdf(-3, 1)(p), bins=51)


def test_multivariate_copula_simple(plt, rng):
    n = 100000

    c = 0.7
    dist = MultivariateCopula(
        [gaussian_icdf(-1, 1), uniform_icdf(-1, 1)], rho=[[1.0, c], [c, 1.0]]
    )
    pts = dist.sample(n, rng=rng)
    assert pts.shape == (n, 2)

    plt.hist2d(pts[:, 0], pts[:, 1], bins=31)


def test_multivariate_gaussian(plt, rng):
    n = 200000

    dist = MultivariateGaussian((-1, 1), (0.25, 4))
    pts = dist.sample(n, rng=rng)

    plt.hist2d(pts[:, 0], pts[:, 1], bins=np.arange(-5, 5, 0.2))


def test_mixture_1d(plt, rng):
    n = 100000
    bins = np.arange(-5, 5, 0.1)

    dists = [Uniform(-0.5, 0.5), Gaussian(-1, 1), Gaussian(2, 1)]
    mdists = [Mixture(dists), Mixture(dists, p=[0.7, 0.15, 0.15])]

    for k, dist in enumerate(mdists):
        plt.subplot(len(mdists), 1, k + 1)
        pts = dist.sample(n, rng=rng)
        plt.hist(pts, bins=bins)


def test_mixture_2d(plt, rng):
    n = 200000

    dists = [
        MultivariateGaussian((-2, -1), (0.25, 4)),
        MultivariateGaussian((1, 1), (1, 1)),
    ]
    mdists = [Mixture(dists), Mixture(dists, p=[0.7, 0.3])]

    for k, dist in enumerate(mdists):
        plt.subplot(len(mdists), 1, k + 1)
        pts = dist.sample(n, d=2, rng=rng)
        plt.hist2d(pts[:, 0], pts[:, 1], bins=np.arange(-5, 5, 0.2))


def test_tile(rng):
    a = rng.uniform(-1, 1, size=(10, 3))

    dist = Tile(a)

    b = dist.sample(25, 4)
    assert np.array_equal(b, np.tile(a, (3, 2))[:25, :4])


@pytest.mark.parametrize(("dimensions"), (1, 8, 32))
@pytest.mark.parametrize(
    ("bounds, mode"), (([-1, 1], 0), ([-0.1, 1], 0.4), ([-0.6, 0.3], 0.2))
)
def test_triangular_area_intercepts(plt, rng, dimensions, bounds, mode):
    n_neurons = 1000

    def analytic_proportion(x, d):
        value = 0.5 * scipy.special.betainc((d + 1) / 2.0, 0.5, 1 - x ** 2)
        if x < 0:
            value = 1.0 - value
        return value

    triangle = Triangular(bounds[0], mode, bounds[1])
    dist = AreaIntercepts(dimensions=dimensions, base=triangle)
    intercepts = dist.sample(n_neurons, rng=rng)

    with nengo.Network() as net:
        ens = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=dimensions, intercepts=intercepts
        )
        pts = ens.eval_points.sample(n=200000, d=ens.dimensions)
    sim = nengo.Simulator(net)
    _, activity = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=pts)
    p = np.mean(activity > 0, axis=0)

    hist = np.histogram(p, bins=20)[0]
    hist2 = np.histogram(
        [analytic_proportion(val, dimensions) for val in intercepts], bins=20
    )[0]

    print(hist)
    print(hist2)

    plt.bar(range(len(hist)), hist)
    plt.bar(range(len(hist2)), hist2, alpha=0.5)
    plt.savefig(
        "dimensions_%i_bounds_%i_%i_%i" % (dimensions, bounds[0], mode, bounds[1])
    )

    assert np.allclose(hist, hist2, rtol=0.2)
