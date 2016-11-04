import nengo
import numpy as np
import pytest
from nengo.dists import Choice
from nengo.processes import WhiteSignal
from nengo.utils.matplotlib import implot
from nengo.utils.numpy import rms

from nengo_extras import FastLIF, SoftLIFRate
from nengo_extras.neurons import rates_isi, rates_kernel


def test_softlifrate_rates(plt):
    gain = 0.9
    bias = 1.7
    tau_rc = 0.03
    tau_ref = 0.002

    lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
    softlif = SoftLIFRate(tau_rc=tau_rc, tau_ref=tau_ref, sigma=0.00001)

    x = np.linspace(-2, 2, 301)
    lif_r = lif.rates(x, gain, bias)
    softlif_r = softlif.rates(x, gain, bias)

    plt.plot(x, lif_r)
    plt.plot(x, softlif_r)

    assert np.allclose(softlif_r, lif_r, atol=1e-3, rtol=1e-3)


def test_softlifrate_derivative(plt):
    sigma = 0.02
    amplitude = 0.8
    gain = 0.9
    bias = 1.7

    x, dx = np.linspace(-2, 2, 301, retstep=True)
    x2 = 0.5 * (x[:-1] + x[1:])

    neuron = SoftLIFRate(sigma=sigma, amplitude=amplitude)
    r = neuron.rates(x, gain, bias)
    deltar = np.diff(r) / dx
    dr = neuron.derivative(x2, gain, bias)

    plt.subplot(211)
    plt.plot(x, r)

    plt.subplot(212)
    plt.plot(x2, dr)
    plt.plot(x2, deltar, 'k--')

    assert np.allclose(dr, deltar, atol=1e-5, rtol=1e-2)


def test_fastlif(plt):
    """Test that the dynamic model approximately matches the rates."""
    # Based nengo.tests.test_neurons.test_lif
    rng = np.random.RandomState(10)

    dt = 1e-3
    n = 5000
    x = 0.5
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=10, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=1, size=n)

    m = nengo.Network()
    with m:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(n, dimensions=1,
                             neuron_type=FastLIF(),
                             encoders=encoders,
                             max_rates=max_rates,
                             intercepts=intercepts)
        nengo.Connection(
            ins, ens.neurons, transform=np.ones((n, 1)), synapse=None)
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, 'voltage')
        ref_probe = nengo.Probe(ens.neurons, 'refractory_time')

    t_final = 1.0
    with nengo.Simulator(m, dt=dt) as sim:
        sim.run(t_final)

    i = 3
    plt.subplot(311)
    plt.plot(sim.trange(), sim.data[spike_probe][:, :i])
    plt.subplot(312)
    plt.plot(sim.trange(), sim.data[voltage_probe][:, :i])
    plt.subplot(313)
    plt.plot(sim.trange(), sim.data[ref_probe][:, :i])
    plt.ylim([-dt, ens.neuron_type.tau_ref + dt])

    # check rates against analytic rates
    math_rates = ens.neuron_type.rates(
        x, *ens.neuron_type.gain_bias(max_rates, intercepts))
    spikes = sim.data[spike_probe]
    sim_rates = (spikes > 0).sum(0) / t_final
    print("ME = %f" % (sim_rates - math_rates).mean())
    print("RMSE = %f" % (
        rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20)))
    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)

    # if voltage and ref time are non-constant, the probe is doing something
    assert np.abs(np.diff(sim.data[voltage_probe])).sum() > 1
    assert np.abs(np.diff(sim.data[ref_probe])).sum() > 1

    # compute spike counts after each timestep
    actual_counts = (spikes > 0).cumsum(axis=0)
    expected_counts = np.outer(sim.trange(), math_rates)
    assert (abs(actual_counts - expected_counts) < 1).all()


def _test_rates(Simulator, rates, plt, seed):
    n = 100
    intercepts = np.linspace(-0.99, 0.99, n)

    model = nengo.Network(seed=seed)
    with model:
        model.config[nengo.Ensemble].max_rates = Choice([50])
        model.config[nengo.Ensemble].encoders = Choice([[1]])
        u = nengo.Node(output=WhiteSignal(2, high=5))
        a = nengo.Ensemble(n, 1,
                           intercepts=intercepts, neuron_type=nengo.LIFRate())
        b = nengo.Ensemble(n, 1,
                           intercepts=intercepts, neuron_type=nengo.LIF())
        nengo.Connection(u, a, synapse=0)
        nengo.Connection(u, b, synapse=0)
        up = nengo.Probe(u)
        ap = nengo.Probe(a.neurons)
        bp = nengo.Probe(b.neurons)

    with Simulator(model, seed=seed+1) as sim:
        sim.run(2.)

    t = sim.trange()
    x = sim.data[up]
    a_rates = sim.data[ap]
    spikes = sim.data[bp]
    b_rates = rates(t, spikes)

    if plt is not None:
        ax = plt.subplot(411)
        plt.plot(t, x)
        ax = plt.subplot(412)
        implot(plt, t, intercepts, a_rates.T, ax=ax)
        ax.set_ylabel('intercept')
        ax = plt.subplot(413)
        implot(plt, t, intercepts, b_rates.T, ax=ax)
        ax.set_ylabel('intercept')
        ax = plt.subplot(414)
        implot(plt, t, intercepts, (b_rates - a_rates).T, ax=ax)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('intercept')

    tmask = (t > 0.1) & (t < 1.9)
    relative_rmse = rms(b_rates[tmask] - a_rates[tmask]) / rms(a_rates[tmask])
    return relative_rmse


def test_rates_isi(Simulator, plt, seed):
    pytest.importorskip('scipy')
    rel_rmse = _test_rates(Simulator, rates_isi, plt, seed)
    assert rel_rmse < 0.3


def test_rates_kernel(Simulator, plt, seed):
    rel_rmse = _test_rates(Simulator, rates_kernel, plt, seed)
    assert rel_rmse < 0.25


@pytest.mark.noassertions
def test_rates(Simulator, seed, logger):
    pytest.importorskip('scipy')
    functions = [
        ('isi_zero', lambda t, s: rates_isi(
            t, s, midpoint=False, interp='zero')),
        ('isi_midzero', lambda t, s: rates_isi(
            t, s, midpoint=True, interp='zero')),
        ('isi_linear', lambda t, s: rates_isi(
            t, s, midpoint=False, interp='linear')),
        ('isi_midlinear', lambda t, s: rates_isi(
            t, s, midpoint=True, interp='linear')),
        ('kernel_expon', lambda t, s: rates_kernel(t, s, kind='expon')),
        ('kernel_gauss', lambda t, s: rates_kernel(t, s, kind='gauss')),
        ('kernel_expogauss', lambda t, s: rates_kernel(
            t, s, kind='expogauss')),
        ('kernel_alpha', lambda t, s: rates_kernel(t, s, kind='alpha')),
    ]

    for name, function in functions:
        rel_rmse = _test_rates(Simulator, function, None, seed)
        logger.info('rate estimator: %s', name)
        logger.info('relative RMSE: %0.4f', rel_rmse)
