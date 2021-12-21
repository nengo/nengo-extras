import nengo
import numpy as np
from nengo.dists import Choice
from nengo.networks.ensemblearray import EnsembleArray


def Product(n_neurons, dimensions, input_magnitude=1, net=None):
    """Computes the element-wise product of two equally sized vectors."""
    if net is None:
        net = nengo.Network(label="Product")

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].encoders = Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    with net, config:
        net.A = nengo.Node(size_in=dimensions, label="A")
        net.B = nengo.Node(size_in=dimensions, label="B")
        net.product = EnsembleArray(
            n_neurons,
            n_ensembles=dimensions,
            ens_dimensions=2,
            radius=input_magnitude * np.sqrt(2),
        )
        nengo.Connection(net.A, net.product.input[::2], synapse=None)
        nengo.Connection(net.B, net.product.input[1::2], synapse=None)
        net.output = net.product.add_output("product", lambda x: x[0] * x[1])
    return net
