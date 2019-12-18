import pytest
import numpy as np

from nengo.dists import Gaussian, Uniform
from nengo_extras.dists import (
    Concatenate,
    gaussian_icdf,
    loggaussian_icdf,
    Mixture,
    MultivariateCopula,
    MultivariateGaussian,
    Tile,
    uniform_icdf,
    generate_triangular,
    Triangular,
    AreaIntercepts
)


def test_concatenate(plt, rng):
    n = 10000

    dist = Concatenate([Uniform(-1, 1),
                        Uniform(0, 1),
                        MultivariateGaussian([0, 2], [2, 1]),
                        Gaussian(3, 0.5)])
    pts = dist.sample(n, rng=rng)
    assert pts.shape == (n, 5)
    n, d = pts.shape

    for i in range(d):
        plt.subplot(d, 1, i+1)
        plt.hist(pts[:, i], bins=np.linspace(-4, 4, 101))


def test_icdfs(plt, rng):
    p = rng.rand(100000)

    rows = 3
    plt.subplot(rows, 1, 1)
    plt.hist(gaussian_icdf(1, 0.5)(p), bins=51)

    plt.subplot(rows, 1, 2)
    h, b = np.histogram(loggaussian_icdf(-2, 0.5, base=10)(p),
                        bins=np.logspace(-4, 0))
    plt.semilogx(0.5*(b[:-1] + b[1:]), h)

    plt.subplot(rows, 1, 3)
    plt.hist(uniform_icdf(-3, 1)(p), bins=51)


def test_multivariate_copula_simple(plt, rng):
    n = 100000

    c = 0.7
    dist = MultivariateCopula([gaussian_icdf(-1, 1),
                               uniform_icdf(-1, 1)],
                              rho=[[1., c], [c, 1.]])
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


def test_tile(rng):
    a = rng.uniform(-1, 1, size=(10, 3))

    dist = Tile(a)

    b = dist.sample(25, 4)
    assert np.array_equal(b, np.tile(a, (3, 2))[:25, :4])


@pytest.mark.parametrize(
    ('n_input, n_ensembles, n_neurons'),(
    (1, 1, 100000),
    (1, 2, 10000)
    )
)
@pytest.mark.parametrize(
    ('bounds, mode'),(
    ([-1, 1], 0),
    ([-1, 1], 0.4),
    ([-0.3, 0.8], 0.4)
    )
)
def test_triangular(plt, n_input, n_ensembles, n_neurons, bounds, mode):
    intercepts = generate_triangular(
        n_input=n_input,
        n_ensembles=n_ensembles,
        n_neurons=n_neurons,
        bounds=bounds,
        mode=mode)

    assert intercepts.shape[0] == n_ensembles
    assert intercepts.shape[1] == n_neurons
    assert (np.asarray(intercepts)>=bounds[0]).all()
    assert (np.asarray(intercepts)<=bounds[1]).all()

    # round to one decimal to speed things up
    # equivalent to having a bin size of 0.1
    n_decimals = 1
    intercepts = np.around(intercepts, n_decimals)
    bin_size = 1/10**n_decimals
    # get a count of the unique intercepts
    for ii in range(n_ensembles):
        vals = np.unique(intercepts[ii])
        plt.figure()
        data = []
        for val in vals:
            count = list(intercepts[ii]).count(val)
            data.append({'val': val, 'count': count})
            plt.scatter(val, count)

        # generate line plot of the expected triangular shape
        # left side
        x1 = np.linspace(bounds[0], mode, len(vals))
        # right side
        x2 = np.linspace(mode, bounds[1], len(vals))
        # max count (should be at mode)
        max_int = bin_size*n_neurons/(0.5*(bounds[1]-bounds[0]))
        # line equations
        y1 = lambda x: (max_int/(mode-bounds[0]))*(x-bounds[0])
        y2 = lambda x: -(max_int/(bounds[1]-mode))*(x-bounds[1])
        plt.plot(x1, y1(x1), 'k', label='expected')
        plt.plot(x2, y2(x2), 'k')
        plt.legend()

        # tolerance of +/- 1.5% total neurons to account for sparse
        # distributions for small networks
        tolerance = 0.015*n_neurons
        for entry in data:
            if entry['val'] < mode:
                y = y1
            else:
                y = y2
            try:
                assert(np.isclose(entry['count'], y(entry['val']),
                    atol=tolerance, rtol=0))
            except AssertionError as e:
                print(
                    'Intercept %f had a count of %f'
                    % (entry['val'], entry['count'])
                    + '\nExcepted: %f +/- %f\nDifference: %f'
                    % (y(entry['val']), tolerance,
                    (entry['count'] - y(entry['val']))))
                raise e
