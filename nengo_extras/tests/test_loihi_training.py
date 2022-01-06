import nengo
import numpy as np
import pytest

from nengo_extras import reqs
from nengo_extras.loihi_training import AlphaRCNoise, LowpassRCNoise, NoiseBuilder
from nengo_extras.neurons import SoftLIFRate

if reqs.HAS_NENGO_DL:
    import nengo_dl

if reqs.HAS_NENGO_LOIHI:
    from nengo_loihi.neurons import (
        LoihiLIF,
        LoihiSpikingRectifiedLinear,
        discretize_tau_rc,
        discretize_tau_ref,
        install_dl_builders,
        loihi_rates,
        nengo_rates,
    )
else:

    LoihiLIF = LoihiSpikingRectifiedLinear = lambda *args, **kwargs: None


def test_noisebuilder_register_twice_warning():
    class MockNoiseBuilder:
        pass

    # First time does not warn
    with pytest.warns(None) as record:
        NoiseBuilder.register(int)(MockNoiseBuilder)
    assert len(record) == 0

    # Second time warns
    with pytest.warns(UserWarning, match="already has a builder"):
        NoiseBuilder.register(int)(MockNoiseBuilder)


@pytest.mark.skipif(not reqs.HAS_NENGO_DL, reason="requires nengo-dl")
@pytest.mark.skipif(not reqs.HAS_NENGO_LOIHI, reason="requires nengo-loihi")
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:Non-finite values detected")
@pytest.mark.parametrize("neuron_type", [LoihiLIF(), LoihiSpikingRectifiedLinear()])
@pytest.mark.parametrize("inference_only", (True, False))
def test_nengo_dl_neurons(neuron_type, inference_only, Simulator, plt, allclose):
    install_dl_builders()

    dt = 0.0007

    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    if isinstance(neuron_type, nengo.SpikingRectifiedLinear):
        bias = np.linspace(0, 1001, n)
    else:
        bias = np.linspace(0, 30, n)

    with nengo.Network() as model:
        nengo_dl.configure_settings(inference_only=inference_only)

        a = nengo.Ensemble(
            n, 1, neuron_type=neuron_type, encoders=encoders, gain=gain, bias=bias
        )
        ap = nengo.Probe(a.neurons)

    t_final = 1.0
    with nengo_dl.Simulator(model, dt=dt) as dl_sim:
        dl_sim.run(t_final)

    with Simulator(model, dt=dt) as loihi_sim:
        loihi_sim.run(t_final)

    rates_dlsim = (dl_sim.data[ap] > 0).sum(axis=0) / t_final
    rates_loihisim = (loihi_sim.data[ap] > 0).sum(axis=0) / t_final

    zeros = np.zeros((1, gain.size))
    rates_ref = neuron_type.rates(zeros, gain, bias, dt=dt).squeeze(axis=0)
    plt.plot(bias, rates_loihisim, "r", label="loihi sim")
    plt.plot(bias, rates_dlsim, "b-.", label="dl sim")
    plt.plot(bias, rates_ref, "k--", label="rates_ref")
    plt.legend(loc="best")

    atol = 1.0 / t_final  # the fundamental unit for our rates
    assert rates_ref.shape == rates_dlsim.shape == rates_loihisim.shape
    assert allclose(rates_dlsim, rates_ref, atol=atol, rtol=0, xtol=1)
    assert allclose(rates_loihisim, rates_ref, atol=atol, rtol=0, xtol=1)


def rate_nengo_dl_net(
    neuron_type, discretize=True, dt=0.001, nx=256, gain=1.0, bias=0.0
):
    """Create a network for determining rate curves with NengoDL.

    Arguments
    ---------
    neuron_type : NeuronType
        The neuron type used in the network's ensemble.
    discretize : bool, optional (Default: True)
        Whether the tau_ref and tau_rc values should be discretized
        before generating rate curves.
    dt : float, optional (Default: 0.001)
        Simulator timestep.
    nx : int, optional (Default: 256)
        Number of x points in the rate curve.
    gain : float, optional (Default: 1.)
        Gain of all neurons.
    bias : float, optional (Default: 0.)
        Bias of all neurons.
    """
    net = nengo.Network()
    net.dt = dt
    net.bias = bias
    net.gain = gain
    lif_kw = dict(amplitude=neuron_type.amplitude)
    if isinstance(neuron_type, LoihiLIF):
        net.x = np.linspace(-1, 30, nx)

        net.sigma = 0.02
        lif_kw["tau_rc"] = neuron_type.tau_rc
        lif_kw["tau_ref"] = neuron_type.tau_ref

        if discretize:
            lif_kw["tau_ref"] = discretize_tau_ref(lif_kw["tau_ref"], dt)
            lif_kw["tau_rc"] = discretize_tau_rc(lif_kw["tau_rc"], dt)
        lif_kw["tau_ref"] += 0.5 * dt

    elif isinstance(neuron_type, LoihiSpikingRectifiedLinear):
        net.x = np.linspace(-1, 999, nx)

        net.tau_ref1 = 0.5 * dt
        net.j = (
            neuron_type.current(net.x[:, np.newaxis], gain, bias).squeeze(axis=1) - 1
        )

    with net:
        if isinstance(neuron_type, LoihiLIF) and discretize:
            nengo_dl.configure_settings(lif_smoothing=net.sigma)

        net.stim = nengo.Node(np.zeros(nx))
        net.ens = nengo.Ensemble(
            nx,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([gain]),
            bias=nengo.dists.Choice([bias]),
        )
        nengo.Connection(net.stim, net.ens.neurons, synapse=None)
        net.probe = nengo.Probe(net.ens.neurons)

    rates = dict(ref=loihi_rates(neuron_type, net.x, gain, bias, dt=dt).squeeze(axis=1))
    # rates['med'] is an approximation of the smoothed Loihi tuning curve
    if isinstance(neuron_type, LoihiLIF):
        rates["med"] = nengo_rates(nengo.LIF(**lif_kw), net.x, gain, bias).squeeze(
            axis=1
        )
    elif isinstance(neuron_type, LoihiSpikingRectifiedLinear):
        rates["med"] = np.zeros_like(net.j)
        rates["med"][net.j > 0] = neuron_type.amplitude / (
            net.tau_ref1 + 1.0 / net.j[net.j > 0]
        )

    return net, rates, lif_kw


@pytest.mark.skipif(not reqs.HAS_NENGO_DL, reason="requires nengo-dl")
@pytest.mark.skipif(not reqs.HAS_NENGO_LOIHI, reason="requires nengo-loihi")
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.parametrize(
    "neuron_type",
    [
        LoihiLIF(amplitude=1.0, tau_rc=0.02, tau_ref=0.002),
        LoihiLIF(amplitude=0.063, tau_rc=0.05, tau_ref=0.001),
        LoihiSpikingRectifiedLinear(),
        LoihiSpikingRectifiedLinear(amplitude=0.42),
    ],
)
def test_nengo_dl_neuron_grads(neuron_type, plt, allclose):
    pytest.importorskip("tensorflow")

    install_dl_builders()

    net, rates, lif_kw = rate_nengo_dl_net(neuron_type)

    with net:
        nengo_dl.configure_settings(
            # run with `training=True`
            learning_phase=True,
            dtype="float64",
        )

    with nengo_dl.Simulator(net, dt=net.dt) as sim:
        sim.run_steps(1, data={net.stim: net.x[None, None, :]})
        y = sim.data[net.probe][0]

    # --- compute spiking rates
    n_spike_steps = 1000
    x_spikes = net.x + np.zeros((1, n_spike_steps, 1), dtype=net.x.dtype)
    with nengo_dl.Simulator(net, dt=net.dt) as sim:
        sim.run_steps(n_spike_steps, data={net.stim: x_spikes})
        y_spikes = sim.data[net.probe]
        y_spikerate = y_spikes.mean(axis=0)

    # --- compute derivatives
    if isinstance(neuron_type, LoihiLIF):
        dy_ref = SoftLIFRate(sigma=net.sigma, **lif_kw).derivative(
            net.x, net.gain, net.bias
        )
    else:
        # use the derivative of rates['med'] (the smoothed Loihi tuning curve)
        dy_ref = np.zeros_like(net.j)
        dy_ref[net.j > 0] = (
            neuron_type.amplitude / (net.j[net.j > 0] * net.tau_ref1 + 1) ** 2
        )

    with nengo_dl.Simulator(net, dt=net.dt) as sim:
        assert sim.unroll == 1
        # note: not actually checking gradients, just using this to get the gradients
        analytic = sim.check_gradients(inputs=[net.x[None, None, :]], atol=1e10)[
            net.probe
        ]["analytic"][0]

        dy = np.diagonal(analytic.copy())

    dx = net.x[1] - net.x[0]
    dy_est = np.diff(nengo.synapses.Alpha(10).filtfilt(rates["ref"], dt=1)) / dx
    x1 = 0.5 * (net.x[:-1] + net.x[1:])

    # --- plots
    plt.subplot(211)
    plt.plot(net.x, rates["med"], "--", label="LIF(tau_ref += 0.5*dt)")
    plt.plot(net.x, y, label="nengo_dl")
    plt.plot(net.x, y_spikerate, label="nengo_dl spikes")
    plt.plot(net.x, rates["ref"], "k--", label="LoihiLIF")
    plt.legend(loc=4)

    plt.subplot(212)
    plt.plot(x1, dy_est, "--", label="diff(smoothed_y)")
    plt.plot(net.x, dy, label="nengo_dl")
    plt.plot(net.x, dy_ref, "k--", label="diff(SoftLIF)")
    plt.legend(loc=1)

    np.fill_diagonal(analytic, 0)
    assert np.all(analytic == 0)

    assert y.shape == rates["ref"].shape
    assert allclose(y, rates["ref"], atol=1e-3, rtol=1e-5)
    assert allclose(dy, dy_ref, atol=1e-3, rtol=1e-5)
    assert allclose(y_spikerate, rates["ref"], atol=1, rtol=1e-2, record_rmse=False)


@pytest.mark.skipif(not reqs.HAS_NENGO_DL, reason="requires nengo-dl")
@pytest.mark.skipif(not reqs.HAS_NENGO_LOIHI, reason="requires nengo-loihi")
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.parametrize(
    "neuron_type",
    [
        LoihiLIF(amplitude=0.3, nengo_dl_noise=LowpassRCNoise(0.001)),
        LoihiLIF(amplitude=0.3, nengo_dl_noise=AlphaRCNoise(0.001)),
    ],
)
def test_nengo_dl_noise(neuron_type, seed, plt, allclose):
    pytest.importorskip("tensorflow")

    install_dl_builders()

    net, rates, _ = rate_nengo_dl_net(neuron_type)
    n_noise = 1000  # number of noise samples per x point

    with net:
        nengo_dl.configure_settings(learning_phase=True)  # run with `training=True`

    with nengo_dl.Simulator(net, dt=net.dt, minibatch_size=n_noise, seed=seed) as sim:
        input_data = {net.stim: np.tile(net.x[None, None, :], (n_noise, 1, 1))}
        sim.step(data=input_data)
        y = sim.data[net.probe][:, 0, :]

    ymean = y.mean(axis=0)
    y25 = np.percentile(y, 25, axis=0)
    y75 = np.percentile(y, 75, axis=0)
    dy25 = y25 - rates["ref"]
    dy75 = y75 - rates["ref"]

    # exponential models roughly fitted to 25/75th percentiles
    x1mask = net.x > 1.5
    x1 = net.x[x1mask]
    if isinstance(neuron_type.nengo_dl_noise, AlphaRCNoise):
        exp_model = 0.7 + 2.8 * np.exp(-0.22 * (x1 - 1))
        atol = 0.12 * exp_model.max()
    elif isinstance(neuron_type.nengo_dl_noise, LowpassRCNoise):
        exp_model = 1.5 + 2.2 * np.exp(-0.22 * (x1 - 1))
        atol = 0.2 * exp_model.max()

    rtol = 0.2
    mu_atol = 0.6  # depends on n_noise and variance of noise

    # --- plots
    plt.subplot(211)
    plt.plot(net.x, rates["med"], "--", label="LIF(tau_ref += 0.5*dt)")
    plt.plot(net.x, ymean, label="nengo_dl")
    plt.plot(net.x, y25, ":", label="25th")
    plt.plot(net.x, y75, ":", label="75th")
    plt.plot(net.x, rates["ref"], "k--", label="LoihiLIF")
    plt.legend()

    plt.subplot(212)
    plt.plot(net.x, ymean - rates["ref"], "b", label="mean")
    plt.plot(net.x, mu_atol * np.ones_like(net.x), "b:")
    plt.plot(net.x, -mu_atol * np.ones_like(net.x), "b:")
    plt.plot(net.x, y25 - rates["ref"], ":", label="25th")
    plt.plot(net.x, y75 - rates["ref"], ":", label="75th")
    plt.plot(x1, exp_model, "k--")
    plt.plot(x1, exp_model * (1 + rtol) + atol, "k:")
    plt.plot(x1, exp_model * (1 - rtol) - atol, "k:")
    plt.plot(x1, -exp_model, "k--")
    plt.legend()

    assert ymean.shape == rates["ref"].shape
    assert allclose(ymean, rates["ref"], atol=mu_atol, record_rmse=False)
    assert allclose(dy25[x1mask], -exp_model, atol=atol, rtol=rtol, record_rmse=False)
    assert allclose(dy75[x1mask], exp_model, atol=atol, rtol=rtol, record_rmse=False)
