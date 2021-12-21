"""Creation of informative spike raster plots.

This module provides the `plot_spikes` function to easily create a spike raster
plot. Often such a plot is more informative with some preprocessing on the
spike trains that selects those with a high variance (i.e., changes in firing
rate) and orders them according to similarity. Such a preprocessing is provided
by the `preprocess_spikes` function. Thus, to create a quick spike raster plot
with such preprocessing::

    plot_spikes(*preprocess_spikes(t, spikes))

Functions for the individual preprocessing steps (and some alternatives) can
also be found in this model to allow to fine-tune the preprocessing according
to the data.
"""

from __future__ import absolute_import

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nengo_extras import reqs

if reqs.HAS_SCIPY:
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.ndimage import gaussian_filter1d

cm_gray_r_a = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gray_r_a", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
)


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

    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("cmap", cm_gray_r_a)
    kwargs.setdefault("interpolation", "nearest")
    kwargs.setdefault("extent", (t[0], t[-1], 0.0, spikes.shape[1]))

    spikeraster = ax.imshow(spikes.T, **kwargs)
    spikeraster.set_clim(0.0, np.max(spikes) * contrast_scale)
    return spikeraster


def preprocess_spikes(
    t,
    spikes,
    num=50,
    sample_size=200,
    sample_filter_width=0.02,
    cluster_filter_width=0.002,
):
    """Applies a default preprocessing to spike data for plotting.

    This will first sample by variance, then cluster the spike trains, and
    finally merge them. See `sample_by_variance`, `cluster`, and `merge` for
    details.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    num : int, optional
        Number of spike trains to return after merging.
    sample_size : int, optional
        Number of spike trains to sample by variance.
    sample_filter_width : float, optional
        Gaussian filter width in seconds for sampling by variance.
    cluster_filter_width : float, optional
        Gaussian filter width in seconds for clustering.

    Returns
    -------
    tuple (t, selected_spikes)
        Returns the time indices *t* and the preprocessed spike trains
        *spikes*.
    """
    return merge(
        *cluster(
            *sample_by_variance(
                t, spikes, num=sample_size, filter_width=sample_filter_width
            ),
            filter_width=cluster_filter_width,
        ),
        num=num,
    )


def cluster(t, spikes, filter_width):
    """Change order of spike trains to have similar ones close together.

    Requires SciPy.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    filter_width : float
        Gaussian filter width in seconds, controls the time scale the
        clustering is sensitive to.

    Returns
    -------
    tuple (t, selected_spikes)
        Returns the time indices *t* and the selected spike trains *spikes*.
    """

    if not reqs.HAS_SCIPY:
        raise ImportError("`cluster` requires `scipy`")

    dt = (t[-1] - t[0]) / (len(t) - 1)
    filtered = gaussian_filter1d(np.asfarray(spikes), filter_width / dt, axis=0)
    order = to_tree(linkage(filtered.T)).pre_order()
    return t, spikes[:, order]


def merge(t, spikes, num):
    spikes = np.asarray(spikes)

    if spikes.shape[1] <= num:
        return t, spikes

    blocksize = int(np.ceil(spikes.shape[1] / num))
    merged = np.array(
        [
            np.mean(spikes[:, (i * blocksize) : ((i + 1) * blocksize)], axis=1)
            for i in range(num)
        ]
    ).T
    return t, merged


def sample_by_variance(t, spikes, num, filter_width):
    """Samples the spike trains with the highest variance.

    Requires SciPy.

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

    if not reqs.HAS_SCIPY:
        raise ImportError("`sample_by_variance` requires `scipy`")

    dt = (t[-1] - t[0]) / (len(t) - 1)
    filtered = gaussian_filter1d(np.asfarray(spikes), filter_width / dt, axis=0)
    selected = np.argsort(np.var(filtered, axis=0))[-1 : (-num - 1) : -1]
    return t, spikes[:, selected]


def sample_by_activity(t, spikes, num, blocksize=None):
    """Samples the spike trains with the highest spiking activity.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    num : int
        Number of spike trains to return.
    blocksize : int, optional
        If not *None*, the spike trains will be divided into blocks of this
        size and the highest activity spike trains are obtained for each block
        individually.

    Returns
    -------
    tuple (t, selected_spikes)
        Returns the time indices *t* and the selected spike trains *spikes*.
    """
    spikes = np.asarray(spikes)

    if spikes.shape[1] <= num:
        return t, spikes

    if blocksize is None:
        blocksize = spikes.shape[1]

    selected = np.empty((len(t), num))
    n_blocks = int(np.ceil(float(spikes.shape[1]) / blocksize))
    n_sel = int(np.ceil(float(num) / n_blocks))
    for i in range(n_blocks):
        block = spikes[:, (i * blocksize) : ((i + 1) * blocksize)]
        activity = np.sum(block, axis=0)
        selected[:, (i * n_sel) : ((i + 1) * n_sel)] = block[
            :, np.argsort(activity)[-1 : (-n_sel - 1) : -1]
        ]

    return t, selected
