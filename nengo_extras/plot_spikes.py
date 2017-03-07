from __future__ import absolute_import

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


cm_gray_r_a = matplotlib.colors.LinearSegmentedColormap.from_list(
    'gray_r_a', [(0., 0., 0., 0.), (0., 0., 0., 1.)])


def plot_spikes(t, spikes, contrast_scale=1.0, ax=None, **kwargs):
    """Plots a spike raster.

    Will use an alpha channel by default which allows to plot colored regions
    below the spike raster to add highlights. You can set the *cmap* keyword
    argument to a different color map like *matplotlib.cm.gray_r* to get an
    opaque spike raster.

    Utilizes Matplotlib's *imshow*.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    contrast_scale : float, optional
        Scales the contrst of the spike raster. This value multiplied with the
        maximum value in *spikes* will determine the minimum *spike* value to
        appear black (or the corresponding color of the chosen colormap).
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. Uses the current axes by default.
    kwargs : dict
        Additional keyword arguments will be passed on to *imshow*.

    Returns
    -------
    matplotlib.image.AxesImage
        The spikeraster.
    """

    t = np.asarray(t)
    spikes = np.asarray(spikes)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('cmap', cm_gray_r_a)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('extent', (t[0], t[-1], 0., spikes.shape[1]))

    spikeraster = ax.imshow(spikes.T, **kwargs)
    spikeraster.set_clim(0., np.max(spikes) * contrast_scale)
    return spikeraster


def sample_by_variance(t, spikes, num, filter_width):
    """Samples the spike trains with the highest variance.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    num : int
        Number of spike trains to return.
    filter_width : float
        Gaussian filter width in seconds, controls the time scale the variance
        calculation is sensitive to.

    Returns
    -------
    tuple (t, selected_spikes)
        Returns the time indices *t* and the selected spike trains *spikes*.
    """

    from scipy.ndimage import gaussian_filter1d
    dt = (t[-1] - t[0]) / (len(t) - 1)
    filtered = gaussian_filter1d(np.asfarray(spikes),
                                 filter_width / dt, axis=0)
    selected = np.argsort(np.var(filtered, axis=0))[-1:(-num - 1):-1]
    return t, spikes[:, selected]
