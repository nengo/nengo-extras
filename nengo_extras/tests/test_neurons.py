import nengo
import numpy as np
from nengo.utils.numpy import rms

from nengo_extras import FastLIF, SoftLIFRate


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
