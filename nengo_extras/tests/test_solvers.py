import nengo
import numpy as np
import pytest

import nengo_extras.data
from nengo_extras.solvers import (
    LstsqClassifier, SoftmaxClassifier, HingeClassifier)


def autodist(X):
    return np.sqrt(((X[:, :, None] - X.T[None, :, :])**2).sum(axis=1))


def _test_classifier(solver, Simulator, seed=None, rng=None):
    n = 100
    din = 50
    dout = 10

    max_dist = 0
    for _ in range(10):
        class_means1 = nengo.dists.UniformHypersphere(surface=True).sample(
            dout, d=din, rng=rng)
        class_dists1 = autodist(class_means1)
        np.fill_diagonal(class_dists1, np.inf)
        dist1 = class_dists1.min()
        if dist1 > max_dist:
            max_dist = dist1
            class_means = class_means1

    def gen(m):
        y = rng.randint(dout, size=m)
        x = class_means[y] + rng.normal(scale=0.25, size=(m, din))
        return x, y

    trainX, trainY = gen(5000)
    testX, testY = gen(1000)
    trainT = nengo_extras.data.one_hot_from_labels(trainY, classes=dout)

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(n, din)
        b = nengo.Node(size_in=dout)
        c = nengo.Connection(
            a, b, eval_points=trainX, function=trainT, solver=solver)

    with Simulator(model) as sim:
        _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=testX)
        outs = np.dot(acts, sim.data[c].weights.T)
        error = (np.argmax(outs, axis=1) != testY).mean()

    assert error < 0.065
    # ^ Threshold chosen based on empirical upper bound for solvers in the
    # repo. Should catch if something breaks.


def test_lstsqclassifier(Simulator, seed, rng):
    solver = LstsqClassifier()
    _test_classifier(solver, Simulator, seed=seed, rng=rng)


def test_softmaxclassifier(Simulator, seed, rng):
    pytest.importorskip('scipy.optimize')
    solver = SoftmaxClassifier()
    _test_classifier(solver, Simulator, seed=seed, rng=rng)


def test_hingeclassifier(Simulator, seed, rng):
    pytest.importorskip('scipy.optimize')
    solver = HingeClassifier(reg=0.02)
    _test_classifier(solver, Simulator, seed=seed, rng=rng)
