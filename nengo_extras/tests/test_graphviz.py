import nengo
import numpy as np
from nengo.utils.builder import remove_passthrough_nodes

from nengo_extras.graphviz import net_diagram, obj_conn_diagram


def make_net():
    with nengo.Network() as model:
        D = 3
        input = nengo.Node([1] * D, label="input")
        a = nengo.networks.EnsembleArray(50, D, label="a")
        b = nengo.networks.EnsembleArray(50, D, label="b")
        output = nengo.Node(None, size_in=D, label="output")

        nengo.Connection(input, a.input, synapse=0.01)
        nengo.Connection(a.output, b.input, synapse=0.01)
        nengo.Connection(b.output, b.input, synapse=0.01, transform=0.9)
        nengo.Connection(a.output, a.input, synapse=0.01, transform=np.ones((D, D)))
        nengo.Connection(b.output, output, synapse=0.01)
    return model


def test_net_diagram():
    """Constructing a .dot file for a network."""
    dot = net_diagram(make_net())
    assert len(dot.splitlines()) == 31


def test_obj_conn_diagram():
    """Constructing a .dot file for a set of objects and connections."""
    model = make_net()
    objs = model.all_nodes + model.all_ensembles
    conns = model.all_connections

    dot = obj_conn_diagram(objs, conns)
    assert len(dot.splitlines()) == 31

    dot = obj_conn_diagram(*remove_passthrough_nodes(objs, conns))
    assert len(dot.splitlines()) == 27
