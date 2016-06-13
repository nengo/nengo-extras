import numpy as np

from nengo.dists import Choice, Uniform, DistributionParam
from nengo.params import FrozenObject, TupleParam
from nengo.utils.compat import range


class Gabor(FrozenObject):
    """Desribes a random generator for Gabor filters."""

    theta = DistributionParam('theta')
    freq = DistributionParam('freq')
    phase = DistributionParam('phase')
    sigma_x = DistributionParam('sigma_x')
    sigma_y = DistributionParam('sigma_y')

    def __init__(self, theta=Uniform(-np.pi, np.pi), freq=Uniform(0.2, 2),
                 phase=Uniform(-np.pi, np.pi),
                 sigma_x=Choice([0.45]), sigma_y=Choice([0.45])):
        self.theta = theta
        self.freq = freq
        self.phase = phase
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def generate(self, n, shape, rng=np.random, norm=1.):
        assert isinstance(shape, tuple) and len(shape) == 2
        thetas = self.theta.sample(n, rng=rng)[:, None, None]
        freqs = self.freq.sample(n, rng=rng)[:, None, None]
        phases = self.phase.sample(n, rng=rng)[:, None, None]
        sigma_xs = self.sigma_x.sample(n, rng=rng)[:, None, None]
        sigma_ys = self.sigma_y.sample(n, rng=rng)[:, None, None]

        x, y = np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0])
        X, Y = np.meshgrid(x, y)

        c, s = np.cos(thetas), np.sin(thetas)
        X1 = X * c + Y * s
        Y1 = -X * s + Y * c

        gabors = np.exp(-0.5 * ((X1 / sigma_xs)**2 + (Y1 / sigma_ys)**2))
        gabors *= np.cos((2 * np.pi) * freqs * X1 + phases)

        if norm is not None:
            gabors *= norm / np.sqrt(
                (gabors**2).sum(axis=(1, 2), keepdims=True)).clip(1e-5, np.inf)

        return gabors


class Mask(FrozenObject):
    """Describes a sparse receptive-field mask for encoders.

    Parameters
    ----------
    image_shape : 2- or 3-tuple
        Shape of the input image, either (height, witdh) or
        (channels, height, width).
    """

    image_shape = TupleParam('image_shape', length=3)

    def __init__(self, image_shape):
        image_shape = ((1,) + tuple(image_shape) if len(image_shape) == 2 else
                       image_shape)
        self.image_shape = image_shape

    def _positions(self, n, shape, rng):
        diff_shape = np.asarray(self.image_shape[1:]) - np.asarray(shape) + 1

        # find random positions for top-left corner of each RF
        i = rng.randint(low=0, high=diff_shape[0], size=n)
        j = rng.randint(low=0, high=diff_shape[1], size=n)
        return i, j

    def generate(self, n, shape, rng=np.random, flatten=False):
        shape = np.asarray(shape)
        assert shape.ndim == 1 and shape.shape[0] == 2

        i, j = self._positions(n, shape, rng)
        mask = np.zeros((n,) + self.image_shape, dtype='bool')
        for k in range(n):
            mask[k, :, i[k]:i[k]+shape[0], j[k]:j[k]+shape[1]] = True

        return mask.reshape(n, -1) if flatten else mask

    def populate(self, filters, rng=np.random, flatten=False):
        filters = np.asarray(filters)
        assert filters.ndim in [3, 4]
        n, shape = filters.shape[0], filters.shape[-2:]
        channels = 1 if filters.ndim == 3 else filters.shape[1]
        assert channels == self.image_shape[0]

        i, j = self._positions(n, shape, rng)
        output = np.zeros((n,) + self.image_shape, dtype=filters.dtype)
        for k in range(n):
            output[k, :, i[k]:i[k]+shape[0], j[k]:j[k]+shape[1]] = filters[k]

        return output.reshape(n, -1) if flatten else output
