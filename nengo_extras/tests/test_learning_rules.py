import nengo
from nengo.exceptions import ValidationError
from nengo.utils.numpy import rms
import numpy as np
import pytest


from nengo_extras.learning_rules import DeltaRule


@pytest.mark.parametrize('post_target', [None, 'in', 'out'])
def test_delta_rule(Simulator, seed, rng, plt, post_target):
    f = np.cos

    learning_rate = 2e-2

    tau_s = 0.005

    t_train = 10  # amount of learning time
    t_test = 3  # amount of testing time

    n = 50
    max_rate = 200
    dmean = 2. / (n * max_rate)  # theoretical mean for on/off decoders
    dr = 2 * 2 * dmean  # twice mean with additional 2x fudge factor
    decoders = rng.uniform(-dr, dr, size=(1, n))

    ens_params = dict(neuron_type=nengo.LIF(),
                      max_rates=nengo.dists.Choice([max_rate]),
                      intercepts=nengo.dists.Uniform(-1, 0.8))

    if post_target == 'in':
        step = lambda j: (j > 1).astype(j.dtype)
        learning_rule_type = DeltaRule(
            learning_rate=learning_rate, post_fn=step, post_target=post_target,
            post_tau=0.005)
    elif post_target == 'out':
        step = lambda s: (s > 18).astype(s.dtype)
        learning_rule_type = DeltaRule(
            learning_rate=learning_rate, post_fn=step, post_target=post_target,
            post_tau=0.005)
    else:
        learning_rule_type = DeltaRule(learning_rate=learning_rate)

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(nengo.processes.WhiteSignal(period=10, high=5))
        a = nengo.Ensemble(n, 1, **ens_params)
        b = nengo.Ensemble(n, 1, **ens_params)
        y = nengo.Node(size_in=1)

        nengo.Connection(u, a, synapse=None)
        nengo.Connection(b.neurons, y, transform=decoders, synapse=tau_s)
        c = nengo.Connection(a.neurons, b.neurons, synapse=tau_s)

        e = nengo.Node(lambda t, x: x if t < t_train else 0, size_in=1)
        eb = nengo.Node(size_in=n)

        nengo.Connection(u, e, transform=-1, function=f,
                         synapse=nengo.synapses.Alpha(tau_s))
        nengo.Connection(b.neurons, e, transform=decoders, synapse=tau_s)
        nengo.Connection(e, eb, synapse=None, transform=decoders.T)

        c.transform = np.zeros((n, n))
        c.learning_rule_type = learning_rule_type
        nengo.Connection(eb, c.learning_rule, synapse=None)

        ep = nengo.Probe(e)
        up = nengo.Probe(u, synapse=nengo.synapses.Alpha(tau_s))
        yp = nengo.Probe(y)

    with Simulator(model, seed=seed+1) as sim:
        sim.run(t_train + t_test)

    t = sim.trange()
    m = t > t_train
    filt = nengo.synapses.Alpha(0.005)
    x = filt.filtfilt(sim.data[up])
    fx = f(x)
    y = filt.filtfilt(sim.data[yp])

    plt.subplot(311)
    plt.plot(t, sim.data[ep])
    plt.ylabel('error')

    plt.subplot(312)
    plt.plot(t, fx)
    plt.plot(t, y)
    plt.ylabel('output')

    plt.subplot(313)
    plt.plot(t[m], fx[m])
    plt.plot(t[m], y[m])
    plt.ylabel('test output')

    plt.tight_layout()

    rms_error = rms(y[m] - fx[m]) / rms(fx[m])
    assert rms_error < 0.3


def test_delta_rule_function_param_size():
    fn = lambda j: j[:-1]
    with pytest.raises(ValidationError):
        DeltaRule(post_fn=fn)
