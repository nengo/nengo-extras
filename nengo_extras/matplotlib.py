from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np


def imshow(
    image, ax=None, vmin=None, vmax=None, invert=False, interpolation="none", axes=False
):
    """Nicer version of Matplotlib's imshow.

    - By default, show the raw image with no interpolation.
    - If the image is greyscale, use grey colormap.
    """
    kwargs = dict(vmin=vmin, vmax=vmax, interpolation=interpolation)

    if image.ndim == 2 or image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0] if image.ndim == 3 else image
        kwargs["cmap"] = "gray" if not invert else "gist_yarg"
    elif image.ndim == 3:
        assert image.shape[2] == 3
        if vmin is not None and vmax is not None:
            image = (image.clip(vmin, vmax) - vmin) / (vmax - vmin)
    else:
        raise ValueError("Wrong number of image dimensions")

    ax = plt.gca() if ax is None else ax
    ax_img = ax.imshow(image, **kwargs)
    if not axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return ax_img


def tile(
    images,
    ax=None,
    rows=9,
    cols=12,
    grid=False,
    gridwidth=1,
    gridcolor="r",
    **show_params,
):
    """Plot a grid of images

    Parameters
    ----------
    images : ndarray (n_images, height, width, channels)
        Array of images to display.
    """
    if images.ndim == 3:
        images = images[:, :, :, None]
    n_images, m, n, nc = images.shape

    if n_images < rows * cols:
        aspect = float(cols) / rows
        rows = int(np.maximum(np.round(np.sqrt(n_images / aspect)), 1))
        cols = int(np.ceil(float(n_images) / rows))

    img_shape = (m * rows, n * cols, nc)
    img = np.zeros(img_shape, dtype=images.dtype)
    for k in range(min(rows * cols, n_images)):
        i, j = k // cols, k % cols
        img[i * m : (i + 1) * m, j * n : (j + 1) * n, :] = images[k]

    ax_img = imshow(img, ax=ax, **show_params)
    ax = ax_img.axes

    if grid:
        for i in range(1, rows):
            ax.plot(
                [-0.5, img.shape[1] - 0.5],
                [i * m - 0.5, i * m - 0.5],
                color=gridcolor,
                linewidth=gridwidth,
            )
        for j in range(1, cols):
            ax.plot(
                [j * n - 0.5, j * n - 0.5],
                [-0.5, img.shape[0] - 0.5],
                color=gridcolor,
                linewidth=gridwidth,
            )

        ax.set_xlim([-0.5, img.shape[1] - 0.5])
        ax.set_ylim([-0.5, img.shape[0] - 0.5])
        ax.invert_yaxis()


def compare(
    image_sets,
    ax=None,
    rows=4,
    cols=12,
    grid=True,
    gridwidth=1,
    gridcolor="r",
    **show_params,
):
    """Compare sets of images in a grid.

    Parameters
    ----------
    image_sets : list of (n_images, height, width, channels) ndarray
        List of the sets of images to compare. Each set of images must be
        the same size.
    """
    d = len(image_sets)

    n_images = image_sets[0].shape[0]
    imshape = image_sets[0].shape[1:]
    m, n = imshape[:2]
    nc = imshape[2] if len(imshape) > 2 else 1
    for q in range(d):
        if image_sets[q].shape != image_sets[0].shape:
            raise ValueError("All image sets must be the same shape as the first")

    if n_images < rows * cols:
        aspect = float(cols) / rows
        rows = int(np.maximum(np.round(np.sqrt(n_images / aspect)), 1))
        cols = int(np.ceil(float(n_images) / rows))

    img_shape = (d * m * rows, n * cols, nc)
    img = np.zeros(img_shape, dtype=image_sets[0].dtype)

    for k in range(min(rows * cols, n_images)):
        i, j = k // cols, k % cols
        for q in range(d):
            img[
                (d * i + q) * m : (d * i + q + 1) * m, j * n : (j + 1) * n
            ] = image_sets[q][k, :].reshape((m, n, nc))

    ax_img = imshow(img, ax=ax, **show_params)
    ax = ax_img.axes

    if grid:
        for i in range(1, rows):
            ax.plot(
                [-0.5, img.shape[1] - 0.5],
                (d * i * m - 0.5) * np.ones(2),
                color=gridcolor,
                linewidth=gridwidth,
            )
        for j in range(1, cols):
            ax.plot(
                [j * n - 0.5, j * n - 0.5],
                [-0.5, img.shape[0] - 0.5],
                color=gridcolor,
                linewidth=gridwidth,
            )

        ax.set_xlim([-0.5, img.shape[1] - 0.5])
        ax.set_ylim([-0.5, img.shape[0] - 0.5])
        ax.invert_yaxis()
