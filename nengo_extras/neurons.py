import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.exceptions import ValidationError
from nengo.params import NumberParam


def softplus(x, sigma=1.):
    x = np.asarray(x)
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
       LIF Neurons. arXiv Preprint, 1510. https://arxiv.org/abs/1510.08829
    """

    sigma = NumberParam('sigma', low=0, low_open=True)

    def __init__(self, sigma=1., **lif_args):
        super(SoftLIFRate, self).__init__(**lif_args)
        self.sigma = sigma  # smoothing around the threshold

    @property
    def _argreprs(self):
        args = super(SoftLIFRate, self)._argreprs
        if self.sigma != 1.:
            args.append("sigma=%s" % self.sigma)
        return args

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        SoftLIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        j = softplus(J - 1, sigma=self.sigma)
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = lif_j(j[j > 0], self.tau_ref, self.tau_rc,
                              amplitude=self.amplitude)

    def derivative(self, x, gain, bias):
        y = gain * x + bias - 1
        j = softplus(y, sigma=self.sigma)
        yy = y[j > 0]
        jj = j[j > 0]
        vv = lif_j(jj, self.tau_ref, self.tau_rc, amplitude=self.amplitude)

        d = np.zeros_like(j)
        d[j > 0] = (gain * self.tau_rc * vv * vv) / (
            self.amplitude * jj * (jj + 1) * (1 + np.exp(-yy / self.sigma)))
        return d


class FastLIF(nengo.neurons.LIF):
    """Faster version of the leaky integrate-and-fire (LIF) neuron model.

    This neuron model is faster than ``LIF`` but does not produce the ideal
    firing rate for larger ``dt`` due to linearization of the tuning curves.
    """

    def step_math(self, dt, J, spiked, voltage, refractory_time):

        # update voltage using accurate exponential integration scheme
        dV = -np.expm1(-dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < self.min_voltage] = self.min_voltage

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and linearly reduce voltage when partway out of ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike (if v > 1 set spiked = 1/dt, else 0)
        spiked[:] = (voltage > 1) / dt

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - 1) / dV[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0
        refractory_time[spiked > 0] = self.tau_ref + spiketime


def spikes2events(t, spikes):
    """Return an event-based representation of spikes (i.e. spike times)"""
    spikes = npext.array(spikes, copy=False, min_dims=2)
    if spikes.ndim > 2:
        raise ValidationError("Cannot handle %d-dimensional arrays"
                              % spikes.ndim, attr='spikes')
    if spikes.shape[-1] != len(t):
        raise ValidationError("Last dimension of 'spikes' must equal 'len(t)'",
                              attr='spikes')

    # find nonzero elements (spikes) in each row, and translate to times
    return [t[spike != 0] for spike in spikes]


def _rates_isi_events(t, events, midpoint, interp):
    import scipy.interpolate

    if len(events) == 0:
        return np.zeros_like(t)

    isis = np.diff(events)

    rt = np.zeros(len(events) + (1 if midpoint else 2))
    rt[1:-1] = 0.5*(events[:-1] + events[1:]) if midpoint else events
    rt[0], rt[-1] = t[0], t[-1]

    r = np.zeros_like(rt)
    r[1:len(isis) + 1] = 1. / isis

    f = scipy.interpolate.interp1d(rt, r, kind=interp, copy=False)
    return f(t)


def rates_isi(t, spikes, midpoint=False, interp='zero'):
    """Estimate firing rates from spikes using ISIs.

    Parameters
    ----------
    t : (M,) array_like
        The times at which raw spike data (spikes) is defined.
    spikes : (M, N) array_like
        The raw spike data from N neurons.
    midpoint : bool, optional
        If true, place interpolation points at midpoints of ISIs. Otherwise,
        the points are placed at the beginning of ISIs.
    interp : string, optional
        Interpolation type, passed to `scipy.interpolate.interp1d` as the
        ``kind`` parameter.

    Returns
    -------
    rates : (M, N) array_like
        The estimated neuron firing rates.
    """
    spike_times = spikes2events(t, spikes.T)
    rates = np.zeros(spikes.shape)
    for i, st in enumerate(spike_times):
        rates[:, i] = _rates_isi_events(t, st, midpoint, interp)

    return rates


def lowpass_filter(x, tau, kind='expon'):
    nt = x.shape[-1]

    if kind == 'expon':
        t = np.arange(0, 5 * tau)
        kern = np.exp(-t / tau) / tau
        delay = tau
    elif kind == 'gauss':
        std = tau / 2.
        t = np.arange(-4 * std, 4 * std)
        kern = np.exp(-0.5 * (t / std)**2) / np.sqrt(2 * np.pi * std**2)
        delay = 4 * std
    elif kind == 'alpha':
        alpha = 1. / tau
        t = np.arange(0, 5 * tau)
        kern = alpha**2 * t * np.exp(-alpha * t)
        delay = tau
    else:
        raise ValidationError("Unrecognized filter kind '%s'" % kind, 'kind')

    delay = int(np.round(delay))
    return np.array(
        [np.convolve(kern, xx, mode='full')[delay:nt + delay] for xx in x])


def rates_kernel(t, spikes, kind='gauss', tau=0.04):
    """Estimate firing rates from spikes using a kernel.

    Parameters
    ----------
    t : (M,) array_like
        The times at which raw spike data (spikes) is defined.
    spikes : (M, N) array_like
        The raw spike data from N neurons.
    kind : str {'expon', 'gauss', 'expogauss', 'alpha'}, optional
        The type of kernel to use. 'expon' is an exponential kernel, 'gauss' is
        a Gaussian (normal) kernel, 'expogauss' is an exponential followed by
        a Gaussian, and 'alpha' is an alpha function kernel.
    tau : float
        The time constant for the kernel. The optimal value will depend on the
        firing rate of the neurons, with a longer tau preferred for lower
        firing rates. The default value of 0.04 works well across a wide range
        of firing rates.
    """
    spikes = spikes.T
    spikes = npext.array(spikes, copy=False, min_dims=2)
    if spikes.ndim > 2:
        raise ValidationError("Cannot handle %d-dimensional arrays"
                              % spikes.ndim, attr='spikes')
    if spikes.shape[-1] != len(t):
        raise ValidationError("Last dimension of 'spikes' must equal 'len(t)'",
                              attr='spikes')

    n, nt = spikes.shape
    dt = t[1] - t[0]

    tau_i = tau / dt
    kind = kind.lower()
    if kind == 'expogauss':
        rates = lowpass_filter(spikes, tau_i, kind='expon')
        rates = lowpass_filter(rates, tau_i / 4, kind='gauss')
    else:
        rates = lowpass_filter(spikes, tau_i, kind=kind)

    return rates.T
