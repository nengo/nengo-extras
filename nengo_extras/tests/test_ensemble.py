import nengo

from nengo_extras.ensemble import tune_ens_parameters


def test_tune_ens_parameters(plt, rng, seed):
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(2, dimensions=1)
        conn = nengo.Connection(ens, ens)
    with nengo.Simulator(net) as sim:
        before = sim.data[conn].solver_info['rmses'][0]

    tune_ens_parameters(ens, rng=rng)

    with nengo.Simulator(net) as sim:
        after = sim.data[conn].solver_info['rmses'][0]

    assert after < before, "RMSE before: %f\tRMSE after: %f" % (before, after)
