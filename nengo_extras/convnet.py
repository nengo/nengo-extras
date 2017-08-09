import numpy as np

from nengo.exceptions import ValidationError
from nengo.processes import Process
from nengo.params import (EnumParam, NdarrayParam, Parameter, ShapeParam,
                          TupleParam, Unconfigurable)
from nengo.utils.compat import is_iterable, range


class Conv2d(Process):
    """Perform 2-D (image) convolution on an input.

    Parameters
    ----------
    shape_in : 3-tuple (n_channels, height, width)
        Shape of the input images: channels, height, width.
    filters : array_like (n_filters, n_channels, f_height, f_width)
        Static filters to convolve with the input. Shape is number of filters,
        number of input channels, filter height, and filter width. Shape can
        also be (n_filters, height, width, n_channels, f_height, f_width)
        to apply different filters at each point in the image, where 'height'
        and 'width' are the input image height and width.
    biases : array_like (1,) or (n_filters,) or (n_filters, height, width)
        Biases to add to outputs. Can have one bias across the entire output
        space, one bias per filter, or a unique bias for each output pixel.
    strides : 2-tuple (vertical, horizontal) or int
        Spacing between filter placements. If an integer
        is provided, the same spacing is used in both dimensions.
    padding : 2-tuple (vertical, horizontal) or int
        Amount of zero-padding around the outside of the input image. Padding
        is applied to both sides, e.g. ``padding=(1, 0)`` will add one pixel
        of padding to the top and bottom, and none to the left and right.
    """

    shape_in = ShapeParam('shape_in', length=3, low=1)
    shape_out = ShapeParam('shape_out', length=3, low=1)
    strides = ShapeParam('strides', length=2, low=1)
    padding = ShapeParam('padding', length=2)
    filters = NdarrayParam('filters', shape=('...',))
    biases = NdarrayParam('biases', shape=('...',), optional=True)

    def __init__(self, shape_in, filters, biases=None, strides=1, padding=0):  # noqa: C901
        self.shape_in = shape_in
        self.filters = filters
        if self.filters.ndim not in [4, 6]:
            raise ValueError(
                "`filters` must have four or six dimensions "
                "(filters, [height, width,] channels, f_height, f_width)")
        if self.filters.shape[-3] != self.shape_in[0]:
            raise ValueError(
                "Filter channels (%d) and input channels (%d) must match"
                % (self.filters.shape[-3], self.shape_in[0]))
        if not all(s % 2 == 1 for s in self.filters.shape[-2:]):
            raise ValueError("Filter shapes must be odd (got %r)"
                             % (self.filters.shape[-2:],))

        self.strides = strides if is_iterable(strides) else [strides] * 2
        self.padding = padding if is_iterable(padding) else [padding] * 2

        nf = self.filters.shape[0]
        nxi, nxj = self.shape_in[1:]
        si, sj = self.filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.strides
        nyi = 1 + max(int(np.ceil(float(2*pi + nxi - si) / sti)), 0)
        nyj = 1 + max(int(np.ceil(float(2*pj + nxj - sj) / stj)), 0)
        self.shape_out = (nf, nyi, nyj)
        if self.filters.ndim == 6 and self.filters.shape[1:3] != (nyi, nyj):
            raise ValueError("Number of local filters %r must match out shape "
                             "%r" % (self.filters.shape[1:3], (nyi, nyj)))

        self.biases = biases if biases is not None else None
        if self.biases is not None:
            if self.biases.size == 1:
                self.biases.shape = (1, 1, 1)
            elif self.biases.size == np.prod(self.shape_out):
                self.biases.shape = self.shape_out
            elif self.biases.size == self.shape_out[0]:
                self.biases.shape = (self.shape_out[0], 1, 1)
            elif self.biases.size == np.prod(self.shape_out[1:]):
                self.biases.shape = (1,) + self.shape_out[1:]
            else:
                raise ValueError(
                    "Biases size (%d) does not match output shape %s"
                    % (self.biases.size, self.shape_out))

        super(Conv2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, shape_in, shape_out, dt, rng):
        assert np.prod(shape_in) == np.prod(self.shape_in)
        assert np.prod(shape_out) == np.prod(self.shape_out)
        shape_in, shape_out = self.shape_in, self.shape_out

        filters = self.filters
        local_filters = filters.ndim == 6
        biases = self.biases

        nc, nxi, nxj = shape_in
        nf, nyi, nyj = shape_out
        si, sj = filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.strides

        def step_conv2d(t, x):
            x = x.reshape(-1, nc, nxi, nxj)
            n = x.shape[0]
            y = np.zeros((n, nf, nyi, nyj), dtype=x.dtype)

            for i in range(nyi):
                for j in range(nyj):
                    i0 = i*sti - pi
                    j0 = j*stj - pj
                    i1, j1 = i0 + si, j0 + sj
                    sli = slice(max(-i0, 0), min(nxi + si - i1, si))
                    slj = slice(max(-j0, 0), min(nxj + sj - j1, sj))
                    w = (filters[:, i, j, :, sli, slj] if local_filters else
                         filters[:, :, sli, slj])
                    xij = x[:, :, max(i0, 0):min(i1, nxi),
                            max(j0, 0):min(j1, nxj)]
                    y[:, :, i, j] = np.dot(
                        xij.reshape(n, -1), w.reshape(nf, -1).T)

            if biases is not None:
                y += biases

            return y.ravel()

        return step_conv2d


class Pool2d(Process):
    """Perform 2-D (image) pooling on an input.

    Parameters
    ----------
    shape_in : 3-tuple (channels, height, width)
        Shape of the input image.
    pool_size : 2-tuple (vertical, horizontal) or int
        Shape of the pooling region. If an integer is provided, the shape will
        be square with the given side length.
    strides : 2-tuple (vertical, horizontal) or int
        Spacing between pooling placements. If ``None`` (default), will be
        equal to ``pool_size`` resulting in non-overlapping pooling.
    kind : "avg" or "max"
        Type of pooling to perform: average pooling or max pooling.
    mode : "full" or "valid"
        If the input image does not divide into an integer number of pooling
        regions, whether to add partial pooling regions for the extra
        pixels ("full"), or discard extra input pixels ("valid").

    Attributes
    ----------
    shape_out : 3-tuple (channels, height, width)
        Shape of the output image.
    """
    shape_in = ShapeParam('shape_in', length=3, low=1)
    shape_out = ShapeParam('shape_out', length=3, low=1)
    pool_size = ShapeParam('pool_size', length=2, low=1)
    strides = ShapeParam('strides', length=2, low=1)
    kind = EnumParam('kind', values=('avg', 'max'))
    mode = EnumParam('mode', values=('full', 'valid'))

    def __init__(self, shape_in, pool_size, strides=None,
                 kind='avg', mode='full'):
        self.shape_in = shape_in
        self.pool_size = (pool_size if is_iterable(pool_size) else
                          [pool_size] * 2)
        self.strides = (strides if is_iterable(strides) else
                        [strides] * 2 if strides is not None else
                        self.pool_size)
        self.kind = kind
        self.mode = mode
        if not all(st <= p for st, p in zip(self.strides, self.pool_size)):
            raise ValueError("Strides %s must be <= pool_size %s" %
                             (self.strides, self.pool_size))

        nc, nxi, nxj = self.shape_in
        nyi_float = float(nxi - self.pool_size[0]) / self.strides[0]
        nyj_float = float(nxj - self.pool_size[1]) / self.strides[1]
        if self.mode == 'full':
            nyi = 1 + int(np.ceil(nyi_float))
            nyj = 1 + int(np.ceil(nyj_float))
        elif self.mode == 'valid':
            nyi = 1 + int(np.floor(nyi_float))
            nyj = 1 + int(np.floor(nyj_float))
        self.shape_out = (nc, nyi, nyj)

        super(Pool2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, shape_in, shape_out, dt, rng):
        assert np.prod(shape_in) == np.prod(self.shape_in)
        assert np.prod(shape_out) == np.prod(self.shape_out)
        nc, nxi, nxj = self.shape_in
        nc, nyi, nyj = self.shape_out
        si, sj = self.pool_size
        sti, stj = self.strides
        kind = self.kind
        nxi2, nxj2 = nyi * sti, nyj * stj

        def step_pool2d(t, x):
            x = x.reshape(-1, nc, nxi, nxj)
            y = np.zeros((x.shape[0], nc, nyi, nyj), dtype=x.dtype)
            n = np.zeros((nyi, nyj))

            for i in range(si):
                for j in range(sj):
                    xij = x[:, :, i:min(nxi2+i, nxi):sti,
                            j:min(nxj2+j, nxj):stj]
                    ni, nj = xij.shape[-2:]
                    if kind == 'max':
                        y[:, :, :ni, :nj] = np.maximum(y[:, :, :ni, :nj], xij)
                    elif kind == 'avg':
                        y[:, :, :ni, :nj] += xij
                        n[:ni, :nj] += 1
                    else:
                        raise NotImplementedError(kind)

            if kind == 'avg':
                y /= n

            return y.ravel()

        return step_pool2d


def softmax(x, axis=None):
    """Stable softmax function"""
    ex = np.exp(x - x.max(axis=axis, keepdims=True))
    return ex / ex.sum(axis=axis, keepdims=True)
