import numpy as np

import nengo
from nengo.processes import Process
from nengo.params import ShapeParam
from nengo.utils.compat import range, is_number


class ARCRouting(Process):

    shape_in = ShapeParam('shape_in', low=1)
    shape_out = ShapeParam('shape_out', low=1)
    axes = ShapeParam('axes')

    def __init__(self, shape_in, shape_out=None, axes=-1):
        tnum = lambda x: (x,) if is_number(x) else x

        self.shape_in = tnum(shape_in)
        self.shape_out = tnum(shape_out) if shape_out is not None else self.shape_in
        d = len(self.shape_in)
        assert len(self.shape_out) == d

        self.axes = tnum(np.arange(d)[list(axes)])
        na = len(self.axes)

        super(ARCRouting, self).__init__(
            default_size_in=np.prod(self.shape_in) + 2*na,
            default_size_out=np.prod(self.shape_out))

    def make_step(self, shape_in, shape_out, dt, rng):
        axes = self.axes
        na = len(axes)
        assert np.prod(shape_in) == np.prod(self.shape_in) + 2*na
        assert np.prod(shape_out) == np.prod(self.shape_out)
        shape_in, shape_out = self.shape_in, self.shape_out
        d = len(self.shape_in)
        nin = np.prod(self.shape_in)

        masks = []
        for axis in axes:
            nx = shape_in[axis]
            ny = shape_out[axis]
            # n2 = (nx - ny) / 2
            n = nx - ny + 1
            mask = np.zeros((nx, ny), dtype='bool')
            for i in range(ny):
                mask[i:i+n, i] = 1

            masks.append(mask)

        min_scale = 1

        def dot_axis(a, w, axis):
            av = np.rollaxis(a, axis, start=a.ndim)  # roll axis to end
            bv = np.dot(av, w)
            return np.rollaxis(bv, -1, start=axis)

        def step_routing(t, x):
            loc = x[-2*na:-na]
            scale = np.maximum(x[-na:], min_scale)
            x = x[:nin].reshape(shape_in)
            y = x

            for k, axis in enumerate(axes):
                nx = shape_in[axis]
                ny = shape_out[axis]
                xi = np.linspace(-0.5*(nx-1), 0.5*(nx-1), nx)
                yi = np.linspace(-0.5*(ny-1), 0.5*(ny-1), ny)

                mu = scale[k]*yi + loc[k]
                sigma = scale[k] / 2.35
                W = masks[k] * np.exp(-(0.5 / sigma**2) * (xi[:, None] - mu)**2)
                W /= W.sum(axis=0, keepdims=True)

                y = dot_axis(y, W, axis)

            return y.ravel()

        return step_routing


def ARCLayer(shape_in, shape_out=None, axes=-1, label='arc'):

    process = ARCRouting(shape_in, shape_out=shape_out, axes=axes)
    nin = np.prod(process.shape_in)
    na = len(process.axes)

    with nengo.Network(label=label) as network:
        network.output = nengo.Node(process, label='%s.output' % label)
        assert network.output.size_in == nin + 2*na
        network.input = nengo.Node(size_in=nin, label='%s.input' % label)
        network.loc = nengo.Node(size_in=na, label='%s.loc' % label)
        network.scale = nengo.Node(size_in=na, label='%s.scale' % label)
        nengo.Connection(network.input, network.output[:nin], synapse=None)
        nengo.Connection(network.loc, network.output[-2*na:-na], synapse=None)
        nengo.Connection(network.scale, network.output[-na:], synapse=None)

    return network
