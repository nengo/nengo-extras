import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError
from nengo.utils.numpy import rms

from nengo_extras.learning_rules import AML, DeltaRule


@pytest.mark.slow
def test_aml(Simulator, seed, rng, plt):
    d = 32
    vocab = nengo.spa.Vocabulary(d, rng=rng)
    n_items = 3
    item_duration = 1.0

    def err_stimulus(t):
        if t <= n_items * item_duration:
            v = vocab.parse("Out" + str(int(t // item_duration))).v
        else:
            v = np.zeros(d)
        return np.concatenate(((1.0, 1.0), v))

    def pre_stimulus(t):
        return vocab.parse("In" + str(int((t // item_duration) % n_items))).v

    with nengo.Network(seed=seed) as model:
        pre = nengo.Ensemble(50 * d, d)
        post = nengo.Node(size_in=d)
        c = nengo.Connection(
            pre, post, learning_rule_type=AML(d), function=lambda x: np.zeros(d)
        )
        err = nengo.Node(err_stimulus)
        inp = nengo.Node(pre_stimulus)
        nengo.Connection(inp, pre)
        nengo.Connection(err, c.learning_rule)
        p_pre = nengo.Probe(pre, synapse=0.01)
        p_post = nengo.Probe(post, synapse=0.01)
        p_err = nengo.Probe(err, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(2 * n_items * item_duration)

    vocab_out = vocab.create_subset(["Out" + str(i) for i in range(n_items)])
    vocab_in = vocab.create_subset(["In" + str(i) for i in range(n_items)])

    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(sim.trange(), nengo.spa.similarity(sim.data[p_pre], vocab_in))
    ax1.set_ylabel(r"Cue $\mathbf{u}(t)$")

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax2.plot(sim.trange(), nengo.spa.similarity(sim.data[p_err][:, 2:], vocab_out))
    ax2.set_ylabel(r"Target $\mathbf{v}(t)$")

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    ax3.plot(sim.trange(), nengo.spa.similarity(sim.data[p_post], vocab_out))
    ax3.set_ylabel("AML output")

    ax1.set_ylim(bottom=0.0)

    for ax in [ax1, ax2, ax3]:
        ax.label_outer()
    fig.tight_layout()

    t = sim.trange()
    similarity = nengo.spa.similarity(sim.data[p_post], vocab_out)
    for i in range(n_items):
        assert item_duration > 0.3
        start = (n_items + i) * item_duration + 0.3
        end = (n_items + i + 1) * item_duration
        assert np.all(similarity[(start < t) & (t <= end), i] > 0.8)


@pytest.mark.parametrize("post_target", [None, "in", "out"])
def test_delta_rule(Simulator, seed, rng, plt, post_target):
    f = np.cos

    learning_rate = 2e-2

    tau_s = 0.005

    t_train = 10  # amount of learning time
    t_test = 3  # amount of testing time

    n = 50
    max_rate = 200
    dmean = 2.0 / (n * max_rate)  # theoretical mean for on/off decoders
    dr = 2 * 2 * dmean  # twice mean with additional 2x fudge factor
    decoders = rng.uniform(-dr, dr, size=(1, n))

    ens_params = dict(
        neuron_type=nengo.LIF(),
        max_rates=nengo.dists.Choice([max_rate]),
        intercepts=nengo.dists.Uniform(-1, 0.8),
    )

    if post_target == "in":
        step = lambda j: (j > 1).astype(j.dtype)
        learning_rule_type = DeltaRule(
            learning_rate=learning_rate,
            post_fn=step,
            post_target=post_target,
            post_tau=0.005,
        )
    elif post_target == "out":
        step = lambda s: (s > 18).astype(s.dtype)
        learning_rule_type = DeltaRule(
            learning_rate=learning_rate,
            post_fn=step,
            post_target=post_target,
            post_tau=0.005,
        )
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

        nengo.Connection(
            u, e, transform=-1, function=f, synapse=nengo.synapses.Alpha(tau_s)
        )
        nengo.Connection(b.neurons, e, transform=decoders, synapse=tau_s)
        nengo.Connection(e, eb, synapse=None, transform=decoders.T)

        c.transform = np.zeros((n, n))
        c.learning_rule_type = learning_rule_type
        nengo.Connection(eb, c.learning_rule, synapse=None)

        ep = nengo.Probe(e)
        up = nengo.Probe(u, synapse=nengo.synapses.Alpha(tau_s))
        yp = nengo.Probe(y)

    with Simulator(model, seed=seed + 1) as sim:
        sim.run(t_train + t_test)

    t = sim.trange()
    m = t > t_train
    filt = nengo.synapses.Alpha(0.005)
    x = filt.filtfilt(sim.data[up])
    fx = f(x)
    y = filt.filtfilt(sim.data[yp])

    plt.subplot(311)
    plt.plot(t, sim.data[ep])
    plt.ylabel("error")

    plt.subplot(312)
    plt.plot(t, fx)
    plt.plot(t, y)
    plt.ylabel("output")

    plt.subplot(313)
    plt.plot(t[m], fx[m])
    plt.plot(t[m], y[m])
    plt.ylabel("test output")

    plt.tight_layout()

    rms_error = rms(y[m] - fx[m]) / rms(fx[m])
    assert rms_error < 0.3


def test_delta_rule_function_param_size():
    fn = lambda j: j[:-1]
    with pytest.raises(ValidationError):
        DeltaRule(post_fn=fn)
