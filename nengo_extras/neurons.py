import numpy as np

import nengo
from nengo.params import NumberParam


def softrelu(x, sigma=1.):
    y = x / sigma
    z = np.array(x)
    z[y < 34.0] = sigma * np.log1p(np.exp(y[y < 34.0]))
    return z
    # ^ 34.0 gives exact answer in 32 or 64 bit but doesn't overflow in 32 bit


def lif_j(j, tau_ref, tau_rc, amplitude=1.):
    return amplitude / (tau_ref + tau_rc * np.log1p(1. / j))


class SoftLIFRate(nengo.neurons.LIFRate):
    """LIF neuron with smoothing around the firing threshold.

    This is a rate version of the LIF neuron whose tuning curve has a
    continuous first derivative, due to the smoothing around the firing
    threshold. It can be used as a substitute for LIF neurons in deep networks
    during training, and then replaced with LIF neurons when running
    the network [1]_.

    Parameters
    ----------
    sigma : float
        Amount of smoothing around the firing threshold. Larger values mean
        more smoothing.
    amplitude : float
        Scaling factor on the output. If 1 (default), output rates correspond
        to LIF neuron model rates.
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.

    References
    ----------
    .. [1] E. Hunsberger & C. Eliasmith (2015). Spiking Deep Networks with
       LIF Neurons. arXiv Preprint, 1510. http://arxiv.org/abs/1510.08829
    """

    sigma = NumberParam('sigma', low=0, low_open=True)
    amplitude = NumberParam('amplitude', low=0, low_open=True)

    def __init__(self, sigma=1., amplitude=1., **lif_args):
        super(SoftLIFRate, self).__init__(**lif_args)
        self.sigma = sigma  # smoothing around the threshold
        self.amplitude = amplitude  # scaling on the output rates

    @property
    def _argreprs(self):
        args = super(SoftLIFRate, self)._argreprs
        if self.sigma != 1.:
            args.append("sigma=%s" % self.sigma)
        if self.amplitude != 1.:
            args.append("amplitude=%s" % self.amplitude)
        return args

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        SoftLIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        j = softrelu(J - 1, sigma=self.sigma)
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = lif_j(j[j > 0], self.tau_ref, self.tau_rc,
                              amplitude=self.amplitude)

    def derivative(self, x, gain, bias):
        y = gain * x + bias - 1
        j = softrelu(y, sigma=self.sigma)
        yy = y[j > 0]
        jj = j[j > 0]
        vv = lif_j(jj, self.tau_ref, self.tau_rc, amplitude=self.amplitude)

        d = np.zeros_like(j)
        d[j > 0] = (gain * self.tau_rc * vv * vv) / (
            self.amplitude * jj * (jj + 1) * (1 + np.exp(-yy / self.sigma)))
        return d
