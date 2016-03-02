import numpy as np

import nengo

from nengo_extras import SoftLIFRate


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
