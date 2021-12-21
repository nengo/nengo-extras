import nengo
import pytest

from nengo_extras.probe import probe_all


def make_net():
    with nengo.Network() as net:
        net.ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        net.node1 = nengo.Node(output=lambda t, x: x, size_in=1)
        net.conn = nengo.Connection(
            net.ens1, net.node1, learning_rule_type=[nengo.PES()]
        )

        with nengo.Network() as net.subnet:
            net.ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
            net.node2 = nengo.Node(output=[0])

    return net


@pytest.mark.parametrize("recursive", [False, True])
def test_probe_all_recursive(recursive):
    net = make_net()
    probes = probe_all(net, recursive=recursive)

    assert len(probes[net.ens1]) == len(net.ens1.probeable)
    # TODO: remove `set` when duplicate "output" bug fixed
    assert len(probes[net.ens1.neurons]) == len(set(net.ens1.neurons.probeable))
    assert len(probes[net.node1]) == len(net.node1.probeable)
    assert len(probes[net.conn]) == len(net.conn.probeable)
    for lr in net.conn.learning_rule:
        assert len(probes[lr]) == len(lr.probeable)

    if recursive:
        assert len(probes) == 8
        assert len(probes[net.ens2]) == len(net.ens2.probeable)
        assert len(probes[net.node2]) == len(net.node2.probeable)
    else:
        assert len(probes) == 5
        assert net.ens2 not in probes
        assert net.ens2.neurons not in probes
        assert net.node2 not in probes


@pytest.mark.parametrize("recursive", [False, True])
def test_probe_all_options(recursive):
    net = make_net()

    probe_all(
        net, recursive=recursive, probe_options={nengo.Ensemble: ["decoded_output"]}
    )

    if recursive:
        assert len(net.probes) == 2
    else:
        assert len(net.probes) == 1


def test_probe_all_kwargs():
    net = make_net()
    probe_all(net, recursive=True, sample_every=0.1, seed=10)

    for probe in net.probes:
        assert probe.sample_every == 0.1
        assert probe.seed == 10
