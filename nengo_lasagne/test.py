import nengo
from nengo.params import Parameter
import numpy as np

import nengo_lasagne


def test_builder():
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    with nengo.Network() as net:
        N = 10
        d = 2
        inp = nengo.Node(output=nengo.processes.PresentInput(inputs, 0.001),
                         label="input")
        ens = nengo.Ensemble(N, d, neuron_type=nengo.RectifiedLinear(),
                             gain=np.ones(N), bias=np.zeros(N), label="ens")
        output = nengo.Node(size_in=1, label="output")

#         nengo.Connection(inp, ens.neurons, transform=np.zeros((N, 2)))
#         nengo.Connection(ens.neurons, output, transform=np.zeros((1, N)))
        nengo.Connection(inp, ens, transform=np.zeros((d, 2)))
        nengo.Connection(ens, output, transform=np.zeros((1, d)))

        p = nengo.Probe(output)

    net.config.configures(nengo_lasagne.Simulator)
    net.config[nengo_lasagne.Simulator].set_param("train_inputs",
                                                  Parameter({inp: inputs}))
    net.config[nengo_lasagne.Simulator].set_param("train_targets",
                                                  Parameter({output: targets}))
    net.config[nengo_lasagne.Simulator].set_param("batch_size", Parameter(4))
    net.config[nengo_lasagne.Simulator].set_param("n_epochs", Parameter(1000))
    net.config[nengo_lasagne.Simulator].set_param("l_rate", Parameter(0.1))

    sim = nengo_lasagne.Simulator(net)
    sim.run_steps(4)

    print sim.data[p]

test_builder()
# test_multidense()
