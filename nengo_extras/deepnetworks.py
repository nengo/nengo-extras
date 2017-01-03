"""
Tools for running deep networks in Nengo

Notes:
- Layers with weights make copies of said weights, both so that they are
  independent (if the model is later updated), and so they are contiguous.
  (Contiguity is required in Nengo when arrays are hashed during build.)
"""

from __future__ import absolute_import

import numpy as np

import nengo
from nengo.params import Default
from nengo.utils.network import with_self

from .convnet import Conv2d, Pool2d, softmax
from .neurons import SoftLIFRate


class Network(nengo.Network):
    @with_self
    def add_data_layer(self, d, name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = DataLayer(d, **kwargs)
        return self.add_layer(layer, inputs=(), name=name)

    @with_self
    def add_neuron_layer(self, n, inputs=None, name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = NeuronLayer(n, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_softmax_layer(self, size, inputs=None, name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = SoftmaxLayer(size, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_dropout_layer(self, size, keep, inputs=None, name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = DropoutLayer(size, keep, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_full_layer(self, weights, biases, inputs=None, name=None,
                       **kwargs):
        kwargs.setdefault('label', name)
        layer = FullLayer(weights, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_local_layer(self, input_shape, filters, biases, inputs=None,
                        name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = LocalLayer(input_shape, filters, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_conv_layer(self, input_shape, filters, biases, inputs=None,
                       name=None, **kwargs):
        kwargs.setdefault('label', name)
        layer = ConvLayer(input_shape, filters, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_pool_layer(self, input_shape, pool_size, inputs=None, name=None,
                       **kwargs):
        kwargs.setdefault('label', name)
        layer = PoolLayer(input_shape, pool_size, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    def compute(self, inputs, output):
        raise NotImplementedError()


class SequentialNetwork(Network):
    def __init__(self, **kwargs):
        super(SequentialNetwork, self).__init__(**kwargs)

        self.layers = []
        self.layers_by_name = {}

    @property
    def input(self):
        return (None if len(self.layers) == 0 else self.layers[0].input)

    @property
    def output(self):
        return (None if len(self.layers) == 0 else self.layers[-1].output)

    @with_self
    def add_layer(self, layer, inputs=None, name=None):
        assert isinstance(layer, Layer)
        assert layer not in self.layers

        if inputs is None:
            inputs = [self.layers[-1]] if self.output is not None else []

        assert len(inputs) == 0 if self.output is None else (
            len(inputs) == 1 and inputs[0] is self.layers[-1])

        for i in inputs:
            nengo.Connection(
                i.output, layer.input, synapse=None, **layer.pre_args)
        self.layers.append(layer)

        if name is not None:
            assert name not in self.layers_by_name
            self.layers_by_name[name] = layer

        return layer

    def layers_to(self, end):
        if end is None:
            return self.layers

        assert end in self.layers
        return self.layers[:self.layers.index(end)+1]

    def compute(self, x, output_layer=None):
        y = x
        for layer in self.layers_to(output_layer):
            y = layer.compute(y)

        return y

    def theano(self, sx, output_layer=None):
        # ensure we have Theano
        import theano

        sy = sx
        for layer in self.layers_to(output_layer):
            sy = layer.theano(sy)

        return sy

    def theano_compute(self, x, output_layer=None, batch_size=256):
        import theano
        import theano.tensor as tt
        sx = tt.matrix(name='x')
        sy = self.theano(sx, output_layer=output_layer)
        f = theano.function([sx], sy, allow_input_downcast=True)

        x = x.reshape((x.shape[0], -1))
        y0 = f(x[:batch_size])
        if len(x) == len(y0):
            return y0
        else:
            assert len(x) > len(y0)
            y = np.zeros((len(x), y0.shape[1]), dtype=y0.dtype)
            y[:batch_size] = y0
            for i in range(batch_size, len(x), batch_size):
                y[i:i+batch_size] = f(x[i:i+batch_size])
            return y


# class TreeNetwork(Network):
#     def __init__(self, **kwargs):
#         super(TreeNetwork, self).__init__(**kwargs)

#         self.inputs = {}  # name -> input object
#         self.outputs = {}  # name -> output object
#         self.layer_inputs = {}  # mapping layer to its input layers
#         self.layers_by_name = {}

#     @property
#     def layers(self):
#         return list(self.layer_inputs)

#     @with_self
#     def add_layer(self, inputs, layer, name=None):
#         assert isinstance(layer, Layer)
#         assert layer not in self.layer_inputs
#         for i in inputs:
#             assert i in self.layer_inputs
#             nengo.Connection(
#                 i.output, layer.input, synapse=None, **layer.pre_args)

#         self.layer_inputs[layer] = tuple(inputs)
#         if name is not None:
#             assert name not in self.layers_by_name
#             self.layers_by_name[name] = layer

#         return layer

#     @with_self
#     def add_named_input(self, name, d, **kwargs):
#         kwargs.setdefault('label', name)
#         layer = DataLayer(d, **kwargs)
#         self.inputs[name] = layer.input
#         return self.add_layer(layer, inputs=(), name=name)

#     def add_named_output(self, name, obj):
#         output = obj.output
#         self.outputs[name] = output
#         return output

#     def compute(self, inputs, output):
#         raise NotImplementedError(
#             "Layers currently only handle one input for compute")

#         assert isinstance(inputs, dict)

#         def as_layer(x):
#             if isinstance(x, str):
#                 return self.layers_by_name[x]
#             else:
#                 assert x in self.layer_inputs
#                 return x

#         if any(isinstance(i, str) for i in inputs):
#             inputs = dict((as_layer(k), v) for k, v in inputs.items())
#         else:
#             assert all(i in self.layer_inputs for i in inputs)
#         output = as_layer(output)

#         layer_inputs = [inputs[i] if i in inputs else self.compute(inputs, i)
#                         for i in self.layer_inputs[output]]
#         assert len(layer_inputs) == 1
#         return output.compute(layer_inputs[0])


class Layer(nengo.Network):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.pre_args = {}  # kwargs for incoming connections
        # self.post_args = {}  # kwargs for outgoing connections

    @property
    def input(self):
        raise NotImplementedError()

    @property
    def output(self):
        raise NotImplementedError()

    @property
    def size_in(self):
        return self.input.size_in

    @property
    def size_out(self):
        return self.output.size_out

    @property
    def shape_in(self):
        return (self.size_in,)

    @property
    def shape_out(self):
        return (self.size_out,)

    def compute(self, x):
        raise NotImplementedError()

    def theano(self, sx):
        raise NotImplementedError()

    def theano_compute(self, x, **kwargs):
        import theano
        import theano.tensor as tt
        sx = tt.matrix()
        f = theano.function([sx], self.theano(sx), **kwargs)
        y = f(x)
        return y

    def _compute_input(self, x):
        x = np.asarray(x)
        if x.ndim == 0:
            raise ValueError("'x' must be at least 1D")
        elif x.ndim == 1:
            assert x.shape[0] == self.size_in
            return x.reshape(1, -1)

        assert x.ndim == 2 and x.shape[1] == self.size_in
        return x


class NodeLayer(Layer):
    def __init__(self, output=None, size_in=None, **kwargs):
        super(NodeLayer, self).__init__(**kwargs)

        with self:
            self.node = nengo.Node(output=output, size_in=size_in)

        if self.label is not None:
            self.node.label = '%s_node' % self.label

    @property
    def input(self):
        return self.node

    @property
    def output(self):
        return self.node


class NeuronLayer(Layer):
    def __init__(self, n, neuron_type=Default, synapse=Default,
                 gain=1., bias=0., amplitude=1., **kwargs):
        super(NeuronLayer, self).__init__(**kwargs)

        with self:
            self.ensemble = nengo.Ensemble(n, 1, neuron_type=neuron_type)
            self.ensemble.gain = _vectorize(gain, n, 'gain')
            self.ensemble.bias = _vectorize(bias, n, 'bias')

            self._output = nengo.Node(size_in=n)
            self.connection = nengo.Connection(
                self.ensemble.neurons, self.output, transform=amplitude,
                synapse=synapse)

        if self.label is not None:
            self.ensemble.label = '%s_ensemble' % self.label
            self.output.label = '%s_output' % self.label

    @property
    def input(self):
        return self.ensemble.neurons

    @property
    def output(self):
        return self._output

    @property
    def neurons(self):
        return self.ensemble.neurons

    @property
    def amplitude(self):
        return self.connection.transform

    @property
    def bias(self):
        return self.ensemble.bias

    @property
    def gain(self):
        return self.ensemble.gain

    @property
    def neuron_type(self):
        return self.ensemble.neuron_type

    @property
    def synapse(self):
        return self.connection.synapse

    def compute(self, x):
        x = self._compute_input(x)
        return self.amplitude * self.neuron_type.rates(x, self.gain, self.bias)

    def theano(self, sx):
        import theano
        import theano.tensor as tt
        floatX = theano.config.floatX
        gain = tt.cast(self.gain, floatX)
        bias = tt.cast(self.bias, floatX)
        amp = tt.cast(self.amplitude, floatX)
        return amp * neuron_theano(self.neuron_type, gain*sx + bias)


class DataLayer(NodeLayer):
    def __init__(self, size, **kwargs):
        super(DataLayer, self).__init__(size_in=size, **kwargs)

    def compute(self, x):
        return x.reshape((x.shape[0], self.size_out))

    def theano(self, sx):
        return sx.reshape((sx.shape[0], self.size_out))


class SoftmaxLayer(NodeLayer):
    def __init__(self, size, **kwargs):
        super(SoftmaxLayer, self).__init__(
            output=lambda t, x: softmax(x), size_in=size, **kwargs)

    def compute(self, x):
        x = self._compute_input(x)
        return softmax(x, axis=-1)

    def theano(self, sx):
        import theano.tensor as tt
        assert sx.ndim == 2
        return tt.nnet.softmax(sx)


class DropoutLayer(NodeLayer):
    def __init__(self, size, keep, **kwargs):
        super(DropoutLayer, self).__init__(size_in=size, **kwargs)
        self.pre_args['transform'] = keep

    @property
    def keep(self):
        return self.pre_args['transform']

    def compute(self, x):
        x = self._compute_input(x)
        return self.keep * x

    def theano(self, sx):
        return self.keep * sx


class FullLayer(NodeLayer):
    def __init__(self, weights, biases, **kwargs):
        assert weights.ndim == 2
        assert biases.size == weights.shape[0]
        super(FullLayer, self).__init__(size_in=weights.shape[0], **kwargs)

        self.weights = np.array(weights)  # copy
        self.biases = np.array(biases)  # copy

        with self:
            self.bias = nengo.Node(output=biases)
            nengo.Connection(self.bias, self.node, synapse=None)

        self.pre_args['transform'] = weights

        if self.label is not None:
            self.bias.label = '%s_bias' % self.label

    def compute(self, x):
        x = self._compute_input(x)
        return np.dot(x, self.weights.T) + self.biases

    def theano(self, sx):
        import theano.tensor as tt
        assert sx.ndim == 2
        return tt.dot(sx, self.weights.T) + self.biases


class ProcessLayer(NodeLayer):
    def __init__(self, process, **kwargs):
        assert isinstance(process, nengo.Process)
        super(ProcessLayer, self).__init__(output=process, **kwargs)
        self._step = None

    @property
    def process(self):
        return self.node.output

    @property
    def shape_in(self):
        return self.process.shape_in

    @property
    def shape_out(self):
        return self.process.shape_out

    def compute(self, x):
        x = self._compute_input(x)
        n = x.shape[0]
        if self._step is None:
            self._step = self.process.make_step(
                self.size_in, self.size_out, dt=1., rng=None)
        return self._step(0, x).reshape(n, -1)


class LocalLayer(ProcessLayer):
    def __init__(self, input_shape, filters, biases,
                 strides=1, padding=0, **kwargs):
        assert filters.ndim == 6
        filters = np.array(filters)  # copy
        biases = np.array(biases)  # copy
        p = Conv2d(input_shape, filters, biases,
                   strides=strides, padding=padding)
        super(LocalLayer, self).__init__(p, **kwargs)

    def theano(self, x):
        import theano.tensor as tt

        filters = self.process.filters
        biases = self.process.biases
        nc, nxi, nxj = self.process.shape_in
        nf, nyi, nyj = self.process.shape_out
        si, sj = self.process.filters.shape[-2:]
        pi, pj = self.process.padding
        sti, stj = self.process.strides

        ys = []
        n = x.shape[0]
        x = x.reshape((n, nc, nxi, nxj))
        for i in range(nyi):
            for j in range(nyj):
                i0 = i*sti - pi
                j0 = j*stj - pj
                i1, j1 = i0 + si, j0 + sj
                sli = slice(max(-i0, 0), min(nxi + si - i1, si))
                slj = slice(max(-j0, 0), min(nxj + sj - j1, sj))
                w = filters[:, i, j, :, sli, slj].reshape(nf, -1)
                xij = x[:, :, max(i0, 0):min(i1, nxi), max(j0, 0):min(j1, nxj)]
                ys.append(tt.dot(xij.reshape((1, n, -1)), w.T))

        y = tt.concatenate(ys, axis=0).reshape((nyi, nyj, n, nf))
        y = tt.transpose(y, (2, 3, 0, 1))

        y += biases
        return y.reshape((n, nf*nyi*nyj))


class ConvLayer(ProcessLayer):
    def __init__(self, input_shape, filters, biases,
                 strides=1, padding=0, border='ceil', **kwargs):
        assert filters.ndim == 4
        filters = np.array(filters)  # copy
        biases = np.array(biases)  # copy
        p = Conv2d(input_shape, filters, biases,
                   strides=strides, padding=padding, border=border)
        super(ConvLayer, self).__init__(p, **kwargs)

    def theano(self, x):
        import theano.tensor as tt

        filters = self.process.filters
        biases = self.process.biases
        nc, nxi, nxj = self.process.shape_in
        nf, nyi, nyj = self.process.shape_out
        si, sj = self.process.filters.shape[-2:]
        pi, pj = self.process.padding
        sti, stj = self.process.strides

        n = x.shape[0]
        x = x.reshape((n, nc, nxi, nxj))

        nxi2 = nyi*sti + si - 2*pi - 1
        nxj2 = nyj*stj + sj - 2*pj - 1
        if self.process.border == 'ceil' and (nxi2 > nxi or nxj2 > nxj):
            xx = tt.zeros((n, nc, nxi2, nxj2), dtype=x.dtype)
            x = tt.set_subtensor(xx[:, :, :nxi, :nxj], x)
        else:
            assert nxi == nxi2 and nxj == nxj2

        y = tt.nnet.conv2d(x, filters,
                           input_shape=(None, nc, nxi2, nxj2),
                           filter_shape=(nf, nc, si, sj),
                           border_mode=(pi, pj),
                           subsample=(sti, stj),
                           filter_flip=False)

        y += biases
        return y.reshape((n, nf*nyi*nyj))


class PoolLayer(ProcessLayer):
    def __init__(self, input_shape, pool_size,
                 strides=None, kind='avg', mode='full', **kwargs):
        p = Pool2d(input_shape, pool_size, strides=strides,
                   kind=kind, mode=mode)
        super(PoolLayer, self).__init__(p, **kwargs)

    def theano(self, x):
        import theano.tensor as tt
        import theano.tensor.signal.pool

        pool_size = self.process.pool_size
        strides = self.process.strides
        nc, nxi, nxj = self.process.shape_in
        nc, nyi, nyj = self.process.shape_out
        mode = dict(max='max', avg='average_exc_pad')[self.process.kind]

        n = x.shape[0]
        x = x.reshape((n, nc, nxi, nxj))
        y = tt.signal.pool.pool_2d(
            x, pool_size, ignore_border=False, st=strides, mode=mode)

        return y.reshape((n, nc*nyi*nyj))


# class NEFLayer(nengo.Network):
#     def __init__(self, n, d_in, d_out, synapse=Default, eval_points=Default,
#                  function=Default, solver=Default,
#                  label=None, seed=None, add_to_container=None, **ens_kwargs):
#         super(nengo.Network, self).__init__(
#             label=label, seed=seed, add_to_container=add_to_container)

#         with self:
#             self.ensemble = nengo.Ensemble(n, d_in, **ens_kwargs)
#             self.output = nengo.Node(size_in=d_out)
#             self.connection = nengo.Connection(
#                 self.ensemble, self.output, synapse=synapse,
#                 eval_points=eval_points, function=function, solver=solver)

#         if label is not None:
#             self.ensemble.label = '%s_ensemble' % label
#             self.output.label = '%s_output' % label

#     @property
#     def input(self):
#         return self.ensemble.neurons

#     @property
#     def amplitude(self):
#         return self.connection.transform

#     @property
#     def neuron_type(self):
#         return self.ensemble.neuron_type

#     @property
#     def synapse(self):
#         return self.connection.synapse


# def _get_network_attr(obj, attr):
#     if isinstance(obj, nengo.Network) and hasattr(obj, attr):
#         return getattr(obj, attr)
#     else:
#         return obj


def _vectorize(value, n, name):
    value = np.asarray(value)
    if value.size == 1:
        return value * np.ones(n)
    elif value.size == n:
        return value
    else:
        raise ValueError("%r must be length 1 or %d (got %d)"
                         % (name, n, value.size))


def neuron_theano(neuron_type, x):
    import theano
    import theano.tensor as tt
    floatX = theano.config.floatX

    if isinstance(neuron_type, nengo.Direct):
        return x
    elif isinstance(neuron_type, nengo.neurons.RectifiedLinear):
        return tt.maximum(x, 0)
    elif isinstance(neuron_type, nengo.neurons.Sigmoid):
        return tt.nnet.sigmoid(x)
    elif isinstance(neuron_type, SoftLIFRate):  # before LIF since subclass
        # do not apply amplitude, since this is done elsewhere!
        tau_ref = tt.cast(neuron_type.tau_ref, floatX)
        tau_rc = tt.cast(neuron_type.tau_rc, floatX)
        sigma = tt.cast(neuron_type.sigma, floatX)
        j = tt.nnet.softplus((x - 1) / sigma) * sigma
        r = 1 / (tau_ref + tau_rc*tt.log1p(1/j))
        return tt.switch(j > 0, r, 0)
    elif isinstance(neuron_type, (nengo.LIF, nengo.LIFRate)):
        tau_ref = tt.cast(neuron_type.tau_ref, floatX)
        tau_rc = tt.cast(neuron_type.tau_rc, floatX)
        j = x - 1
        r = 1. / (tau_ref + tau_rc*tt.log1p(1./j))
        return tt.switch(j > 0, r, 0)
    else:
        raise NotImplementedError("Neuron type %r" % neuron_type)
