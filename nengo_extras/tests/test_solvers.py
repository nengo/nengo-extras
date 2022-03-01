import pytest

from nengo.tests.test_solvers import get_rate_function, get_encoders, get_eval_points
import numpy as np

from nengo_extras.solvers import Dales, DalesL2


@pytest.mark.parametrize('Solver', [Dales, DalesL2])
def test_dales(Solver, plt, rng):
    dims = 2
    a_neurons, b_neurons = 100, 101
    n_points = 1000

    rates = get_rate_function(a_neurons, dims, rng=rng)
    Ea = get_encoders(a_neurons, dims, rng=rng)  # pre encoders
    Eb = get_encoders(b_neurons, dims, rng=rng)  # post encoders

    train = get_eval_points(n_points, dims, rng=rng)  # training eval points
    Atrain = rates(np.dot(train, Ea))  # training activations
    Xtrain = train  # training targets

    # find decoders and multiply by encoders to get weights
    D, _ = Solver()(Atrain, Xtrain, rng=rng)
    W1 = np.dot(D, Eb)

    # find weights directly
    W2, _ = Solver(weights=True)(Atrain, Xtrain, rng=rng, E=Eb)

    # assert that post inputs are close on test points
    test = get_eval_points(n_points, dims, rng=rng)  # testing eval points
    Atest = rates(np.dot(test, Ea))
    Y1 = np.dot(Atest, W1)  # post inputs from decoders
    Y2 = np.dot(Atest, W2)  # post inputs from weights
    assert np.allclose(Y1, Y2)

    # assert that weights themselves are close (this is true for L2 weights)

    assert np.allclose(W1, W2)
