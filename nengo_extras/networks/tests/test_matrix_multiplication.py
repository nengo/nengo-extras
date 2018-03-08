import nengo
import numpy as np

import nengo_extras


def test_matrix_mult(Simulator, rng, nl, plt):
    shape_left = (2, 2)
    shape_right = (2, 2)

    left_mat = rng.rand(*shape_left)
    right_mat = rng.rand(*shape_right)

    with nengo.Network("Matrix multiplication test") as model:
        node_left = nengo.Node(left_mat.ravel())
        node_right = nengo.Node(right_mat.ravel())

        with nengo.Config(nengo.Ensemble) as cfg:
            cfg[nengo.Ensemble].neuron_type = nl()
            mult_net = nengo_extras.networks.MatrixMult(
                100, shape_left, shape_right)

        p = nengo.Probe(mult_net.output, synapse=0.01)

        nengo.Connection(node_left, mult_net.input_left)
        nengo.Connection(node_right, mult_net.input_right)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1)

    t = sim.trange()
    plt.plot(t, sim.data[p])
    for d in np.dot(left_mat, right_mat).flatten():
        plt.axhline(d, color='k')

    atol, rtol = .2, .01
    ideal = np.dot(left_mat, right_mat).ravel()
    assert np.allclose(sim.data[p][-1], ideal, atol=atol, rtol=rtol)
