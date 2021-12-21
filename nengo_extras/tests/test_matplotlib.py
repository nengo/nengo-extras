import numpy as np
import pytest

import nengo_extras.matplotlib as ne_plt


def wave_image(shape, angle, color=None):
    assert len(shape) == 2

    i = np.linspace(-1, 1, shape[0])
    j = np.linspace(-1, 1, shape[1])
    I, J = np.meshgrid(i, j, indexing="ij")

    c, s = np.cos(angle), np.sin(angle)
    # X = c*I + s*J
    X = -c * J + s * I

    image = 0.5 * np.cos(2 * np.pi * X) + 0.5

    if color is not None:
        color = np.asarray(color)
        assert color.size == 3
        image = image[:, :, None] * color[None, None, :]

    return image


def test_imshow_gray(plt):
    image = wave_image((31, 32), 0.2)
    ne_plt.imshow(image)


def test_imshow_color(plt):
    image = wave_image((31, 32), 0.6, color=(1, 1, 0))
    ne_plt.imshow(image)


def test_imshow_color_vlim(plt):
    image = wave_image((31, 32), 0.6, color=(1, 1, 0))
    image = 2 * image - 1
    ne_plt.imshow(image, vmin=-1, vmax=1)


@pytest.mark.parametrize("color", (False, True))
def test_tile(plt, rng, color):
    n = 200
    angles = rng.uniform(0, np.pi, size=n)
    colors = rng.uniform(size=(n, 3))

    genimg = lambda k: wave_image(
        (30, 31), angles[k], color=colors[k] if color else None
    )
    images = np.array([genimg(k) for k in range(n)])

    ne_plt.tile(images, grid=True, gridcolor=(1, 1, 0))


@pytest.mark.parametrize("color", (False, True))
def test_compare(plt, rng, color):
    d = 3
    n = 10
    angles = rng.uniform(0, np.pi, size=n)
    colors = rng.uniform(size=(n, 3))

    genimg = lambda k: wave_image(
        (30, 31), angles[k] + rng.uniform(-0.3, 0.3), color=colors[k] if color else None
    )
    image_sets = [np.array([genimg(k) for k in range(n)]) for _ in range(d)]

    ne_plt.compare(image_sets)
