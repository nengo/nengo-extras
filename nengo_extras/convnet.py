import numpy as np

from nengo.processes import Process
from nengo.params import EnumParam, IntParam, NdarrayParam, TupleParam
from nengo.utils.compat import range


class Conv2d(Process):
    """Perform 2-D (image) convolution on an input.

    Parameters
    ----------
    filters : array_like (n_filters, n_channels, f_height, f_width)
        Static filters to convolve with the input. Shape is number of filters,
        number of input channels, filter height, and filter width. Shape can
        also be (n_filters, height, width, n_channels, f_height, f_width)
        to apply different filters at each point in the image, where 'height'
        and 'width' are the input image height and width.
    shape_in : 3-tuple (n_channels, height, width)
        Shape of the input images: channels, height, width.
    """

    shape_in = TupleParam('shape_in', length=3)
    shape_out = TupleParam('shape_out', length=3)
    stride = TupleParam('stride', length=2)
    padding = TupleParam('padding', length=2)
    filters = NdarrayParam('filters', shape=('...',))
    biases = NdarrayParam('biases', shape=('...',), optional=True)

    def __init__(self, shape_in, filters, biases=None, stride=1, padding=0):  # noqa: C901
        from nengo.utils.compat import is_iterable, is_integer

        self.shape_in = tuple(shape_in)
        if len(self.shape_in) != 3:
            raise ValueError("`shape_in` must have three dimensions "
                             "(channels, height, width)")

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

        self.stride = stride if is_iterable(stride) else [stride] * 2
        if not all(is_integer(s) and s >= 1 for s in self.stride):
            raise ValueError("All strides must be integers >= 1 (got %s)"
                             % (self.stride,))

        self.padding = padding if is_iterable(padding) else [padding] * 2
        if not all(is_integer(p) and p >= 0 for p in self.padding):
            raise ValueError("All padding must be integers >= 0 (got %s)"
                             % (self.padding,))

        nf = self.filters.shape[0]
        nxi, nxj = self.shape_in[1:]
        si, sj = self.filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.stride
        nyi = 1 + max(int(np.ceil((2*pi + nxi - si) / float(sti))), 0)
        nyj = 1 + max(int(np.ceil((2*pj + nxj - sj) / float(stj))), 0)
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

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        filters = self.filters
        local_filters = filters.ndim == 6
        biases = self.biases
        shape_in = self.shape_in
        shape_out = self.shape_out

        nxi, nxj = shape_in[-2:]
        nyi, nyj = shape_out[-2:]
        nf = filters.shape[0]
        si, sj = filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.stride

        def step_conv2d(t, x):
            x = x.reshape(shape_in)
            y = np.zeros(shape_out)

            for i in range(nyi):
                for j in range(nyj):
                    i0 = i*sti - pi
                    j0 = j*stj - pj
                    i1, j1 = i0 + si, j0 + sj
                    sli = slice(max(-i0, 0), min(nxi + si - i1, si))
                    slj = slice(max(-j0, 0), min(nxj + sj - j1, sj))
                    w = (filters[:, i, j, :, sli, slj] if local_filters else
                         filters[:, :, sli, slj])
                    xij = x[:, max(i0, 0):min(i1, nxi),
                            max(j0, 0):min(j1, nxj)]
                    y[:, i, j] = np.dot(w.reshape(nf, -1), xij.ravel())

            if biases is not None:
                y += biases

            return y.ravel()

        return step_conv2d


class Pool2d(Process):
    """Perform 2-D (image) pooling on an input."""
    shape_in = TupleParam('shape_in', length=3)
    shape_out = TupleParam('shape_out', length=3)
    size = IntParam('size', low=1)
    stride = IntParam('stride', low=1)
    kind = EnumParam('kind', values=('avg', 'max'))

    def __init__(self, shape_in, size, stride=None, kind='avg'):
        self.shape_in = shape_in
        self.size = size
        self.stride = stride if stride is not None else size
        self.kind = kind
        if self.stride > self.size:
            raise ValueError("Stride (%d) must be <= size (%d)" %
                             (self.stride, self.size))

        nc, nxi, nxj = self.shape_in
        nyi = 1 + int(np.ceil(float(nxi - size) / self.stride))
        nyj = 1 + int(np.ceil(float(nxj - size) / self.stride))
        self.shape_out = (nc, nyi, nyj)

        super(Pool2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)
        nc, nxi, nxj = self.shape_in
        nc, nyi, nyj = self.shape_out
        s = self.size
        st = self.stride
        kind = self.kind
        nxi2, nxj2 = nyi * st, nyj * st

        def step_pool2d(t, x):
            x = x.reshape(nc, nxi, nxj)
            y = np.zeros((nc, nyi, nyj), dtype=x.dtype)
            n = np.zeros((nyi, nyj))

            for i in range(s):
                for j in range(s):
                    xij = x[:, i:min(nxi2+i, nxi):st, j:min(nxj2+j, nxj):st]
                    ni, nj = xij.shape[-2:]
                    if kind == 'max':
                        y[:, :ni, :nj] = np.maximum(y[:, :ni, :nj], xij)
                    elif kind == 'avg':
                        y[:, :ni, :nj] += xij
                        n[:ni, :nj] += 1
                    else:
                        raise NotImplementedError(kind)

            if kind == 'avg':
                y /= n

            return y.ravel()

        return step_pool2d
