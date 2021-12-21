import nengo.utils.numpy as npext
import numpy as np
from nengo.dists import Choice, DistributionParam, Uniform
from nengo.params import FrozenObject, TupleParam


class Gabor(FrozenObject):
    """Describes a random generator for Gabor filters."""

    theta = DistributionParam("theta")
    freq = DistributionParam("freq")
    phase = DistributionParam("phase")
    sigma_x = DistributionParam("sigma_x")
    sigma_y = DistributionParam("sigma_y")

    def __init__(
        self,
        theta=Uniform(-np.pi, np.pi),
        freq=Uniform(0.2, 2),
        phase=Uniform(-np.pi, np.pi),
        sigma_x=Choice([0.45]),
        sigma_y=Choice([0.45]),
    ):
        super().__init__()
        self.theta = theta
        self.freq = freq
        self.phase = phase
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def generate(self, n, shape, rng=np.random, norm=1.0):
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

        gabors = np.exp(-0.5 * ((X1 / sigma_xs) ** 2 + (Y1 / sigma_ys) ** 2))
        gabors *= np.cos((2 * np.pi) * freqs * X1 + phases)

        if norm is not None:
            gabors *= norm / np.sqrt(
                (gabors ** 2).sum(axis=(1, 2), keepdims=True)
            ).clip(1e-5, np.inf)

        return gabors


class Mask(FrozenObject):
    """Describes a sparse receptive-field mask for encoders.

    Parameters
    ----------
    image_shape : 2- or 3-tuple
        Shape of the input image, either (height, width) or
        (channels, height, width).
    """

    image_shape = TupleParam("image_shape", length=3)

    def __init__(self, image_shape):
        super().__init__()
        image_shape = (
            (1,) + tuple(image_shape) if len(image_shape) == 2 else image_shape
        )
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
        mask = np.zeros((n,) + self.image_shape, dtype="bool")
        for k in range(n):
            mask[k, :, i[k] : i[k] + shape[0], j[k] : j[k] + shape[1]] = True

        return mask.reshape((n, -1)) if flatten else mask

    def populate(self, filters, rng=np.random, flatten=False):
        filters = np.asarray(filters)
        assert filters.ndim in [3, 4]
        n, shape = filters.shape[0], filters.shape[-2:]
        channels = 1 if filters.ndim == 3 else filters.shape[1]
        assert channels == self.image_shape[0]

        i, j = self._positions(n, shape, rng)
        output = np.zeros((n,) + self.image_shape, dtype=filters.dtype)
        for k in range(n):
            output[k, :, i[k] : i[k] + shape[0], j[k] : j[k] + shape[1]] = filters[k]

        return output.reshape((n, -1)) if flatten else output


def ciw_encoders(
    n_encoders,
    trainX,
    trainY,
    rng=np.random,
    normalize_data=True,
    normalize_encoders=True,
):
    """Computed Input Weights (CIW) method for encoders from data.

    Parameters
    ----------
    n_encoders : int
        Number of encoders to generate.
    trainX : (n_samples, n_dimensions) array-like
        Training features.
    trainY : (n_samples,) array-like
        Training labels.

    Returns
    -------
    encoders : (n_encoders, n_dimensions) array
        Generated encoders.

    References
    ----------
    .. [1] McDonnell, M. D., Tissera, M. D., Vladusich, T., Van Schaik, A.,
       Tapson, J., & Schwenker, F. (2015). Fast, simple and accurate
       handwritten digit classification by training shallow neural network
       classifiers with the "Extreme learning machine" algorithm. PLoS ONE,
       10(8), 1-20. doi:10.1371/journal.pone.0134254
    """
    assert trainX.shape[0] == trainY.size
    trainX = trainX.reshape((trainX.shape[0], -1))
    trainY = trainY.ravel()
    classes = np.unique(trainY)

    assert n_encoders % len(classes) == 0
    n_enc_per_class = n_encoders / len(classes)

    # normalize
    if normalize_data:
        trainX = (trainX - trainX.mean()) / trainX.std()
        # trainX = (trainX - trainX.mean(axis=0)) / trainX.std()
        # trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0) + 1e-8)

    # generate
    encoders = []
    for label in classes:
        X = trainX[trainY == label]
        plusminus = rng.choice([-1, 1], size=(X.shape[0], n_enc_per_class))
        samples = np.dot(plusminus.T, X)
        encoders.append(samples)

    encoders = np.vstack(encoders)
    if normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    return encoders


def cd_encoders_biases(
    n_encoders, trainX, trainY, rng=np.random, mask=None, norm_min=0.05, norm_tries=10
):
    """Constrained difference (CD) method for encoders from data.

    Parameters
    ----------
    n_encoders : int
        Number of encoders to generate.
    trainX : (n_samples, n_dimensions) array-like
        Training features.
    trainY : (n_samples,) array-like
        Training labels.

    Returns
    -------
    encoders : (n_encoders, n_dimensions) array
        Generated encoders.
    biases : (n_encoders,) array
        Generated biases. These are biases assuming ``f = G[E * X + b]``,
        and are therefore more like Nengo's ``intercepts``.

    References
    ----------
    .. [1] McDonnell, M. D., Tissera, M. D., Vladusich, T., Van Schaik, A.,
       Tapson, J., & Schwenker, F. (2015). Fast, simple and accurate
       handwritten digit classification by training shallow neural network
       classifiers with the "Extreme learning machine" algorithm. PLoS ONE,
       10(8), 1-20. doi:10.1371/journal.pone.0134254
    """
    assert trainX.shape[0] == trainY.size
    trainX = trainX.reshape((trainX.shape[0], -1))
    trainY = trainY.ravel()
    d = trainX.shape[1]
    classes = np.unique(trainY)
    assert mask is None or mask.shape == (n_encoders, d)

    inds = [(trainY == label).nonzero()[0] for label in classes]
    train_norm = npext.norm(trainX, axis=1).mean()

    encoders = np.zeros((n_encoders, d))
    biases = np.zeros(n_encoders)
    for k in range(n_encoders):
        for _ in range(norm_tries):
            i, j = rng.choice(len(classes), size=2, replace=False)
            a, b = trainX[rng.choice(inds[i])], trainX[rng.choice(inds[j])]
            dab = a - b
            if mask is not None:
                dab *= mask[k]
            ndab = npext.norm(dab) ** 2
            if ndab >= norm_min * train_norm:
                break
        else:
            raise ValueError("Cannot find valid encoder")

        encoders[k] = (2.0 / ndab) * dab
        biases[k] = np.dot(a + b, dab) / ndab

    return encoders, biases


def percentile_biases(encoders, trainX, percentile=50):
    """Pick biases such that neurons are active for a percentile of inputs."""
    trainX = trainX.reshape((trainX.shape[0], -1))
    H = np.dot(trainX, encoders.T)
    biases = np.percentile(H, percentile, axis=0)
    return biases
