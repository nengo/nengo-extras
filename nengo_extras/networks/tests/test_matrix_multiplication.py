import nengo
import numpy as np

import nengo_extras


def test_matrix_mult(Simulator, seed, rng, AnyNeuronType, plt, allclose):
    shape_left = (2, 2)
    shape_right = (2, 2)

    left_mat = rng.uniform(-1, 1, size=shape_left)
    right_mat = rng.uniform(-1, 1, size=shape_right)

    with nengo.Network("Matrix multiplication test", seed=seed) as model:
        node_left = nengo.Node(left_mat.ravel())
        node_right = nengo.Node(right_mat.ravel())

        with nengo.Config(nengo.Ensemble) as cfg:
            cfg[nengo.Ensemble].neuron_type = AnyNeuronType()
            mult_net = nengo_extras.networks.MatrixMult(200, shape_left, shape_right)

        p = nengo.Probe(mult_net.output, synapse=0.01)

        nengo.Connection(node_left, mult_net.input_left)
        nengo.Connection(node_right, mult_net.input_right)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1)

    t = sim.trange()
    plt.plot(t, sim.data[p])
    for d in np.dot(left_mat, right_mat).flatten():
        plt.axhline(d, color="k")

    atol, rtol = 0.15, 0.01
    if AnyNeuronType.__name__ == "SpikingTanh":
        atol = 0.2
    ideal = np.dot(left_mat, right_mat).ravel()
    assert allclose(sim.data[p][-1], ideal, atol=atol, rtol=rtol)
