import matplotlib.patches
import nengo
import numpy as np
import pytest

from nengo_extras.plot_spikes import plot_spikes


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
