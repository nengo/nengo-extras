import matplotlib.patches
import nengo
import numpy as np
import pytest

from nengo_extras.plot_spikes import (
    plot_spikes, sample_by_activity, sample_by_variance)


@pytest.mark.noassertions
def test_plot_spikes(plt, seed, RefSimulator):
    with nengo.Network(seed=seed) as model:
        ens = nengo.Ensemble(10, 1)
        inp = nengo.Node(np.sin)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens.neurons, 'spikes')

    with RefSimulator(model) as sim:
        sim.run(1.)

    ax = plt.gca()
    ax.add_patch(matplotlib.patches.Rectangle(
        (0, 0), 1, 10,
        fc=(0.8, 0.6, 0.6)))  # To check for transparency
    plot_spikes(sim.trange(), sim.data[p], ax=ax, zorder=1)


def test_sample_by_variance():
    dt = 0.001
    t = np.arange(0., 1., dt) + dt

    spikes = np.zeros((len(t), 4))
    spikes[::, 1] = 1. / dt
    spikes[::10, 2] = 1. / dt
    spikes[::100, 3] = 1. / dt

    t_sampled, spikes_sampled = sample_by_variance(
        t, spikes, num=1, filter_width=0.001)
    assert (t_sampled == t).all()
    assert (spikes_sampled == spikes[:, [2]]).all()

    t_sampled, spikes_sampled = sample_by_variance(
        t, spikes, num=1, filter_width=0.1)
    assert (t_sampled == t).all()
    assert (spikes_sampled == spikes[:, [3]]).all()

    t_sampled, spikes_sampled = sample_by_variance(
        t, spikes, num=2, filter_width=0.1)
    assert (t_sampled == t).all()
    assert (spikes_sampled == spikes[:, [3, 2]]).all()

    t_sampled, spikes_sampled = sample_by_variance(
        t, spikes, num=20, filter_width=0.1)
    assert (t_sampled == t).all()
    assert (spikes_sampled == spikes[:, [3, 2, 1, 0]]).all()


def test_sample_by_activity(plt):
    dt = 0.001
    t = np.arange(0., 1., dt) + dt

    spikes = np.zeros((len(t), 4))
    spikes[:, 1::2] = 1. / dt

    t_sampled, spikes_sampled = sample_by_activity(t, spikes, num=2)
    assert (t_sampled == t).all()
    assert (spikes_sampled == np.ones((len(t), 2)) / dt).all()

    t_sampled, spikes_sampled = sample_by_activity(
        t, spikes, num=1, blocksize=2)
    assert (t_sampled == t).all()
    assert (spikes_sampled == np.ones((len(t), 1)) / dt).all()

    t_sampled, spikes_sampled = sample_by_activity(
        t, spikes, num=2, blocksize=2)
    assert (t_sampled == t).all()
    assert (spikes_sampled == np.ones((len(t), 2)) / dt).all()
