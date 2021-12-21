"""
Tools for running deep networks in Nengo

Notes:
- Layers with weights make copies of said weights, both so that they are
  independent (if the model is later updated), and so they are contiguous.
  (Contiguity is required in Nengo when arrays are hashed during build.)
"""

from __future__ import absolute_import

import nengo
import numpy as np
from nengo.params import Default
from nengo.utils.network import with_self

from nengo_extras.convnet import Conv2d, Pool2d, softmax


class DeepNetwork(nengo.Network):
    @with_self
    def add_data_layer(self, d, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = DataLayer(d, **kwargs)
        return self.add_layer(layer, inputs=(), name=name)

    @with_self
    def add_neuron_layer(self, n, inputs=None, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = NeuronLayer(n, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_softmax_layer(self, size, inputs=None, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = SoftmaxLayer(size, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_dropout_layer(self, size, keep, inputs=None, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = DropoutLayer(size, keep, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_full_layer(self, weights, biases, inputs=None, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = FullLayer(weights, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_local_layer(
        self, input_shape, filters, biases, inputs=None, name=None, **kwargs
    ):
        kwargs.setdefault("label", name)
        layer = LocalLayer(input_shape, filters, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_conv_layer(
        self, input_shape, filters, biases, inputs=None, name=None, **kwargs
    ):
        kwargs.setdefault("label", name)
        layer = ConvLayer(input_shape, filters, biases, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    @with_self
    def add_pool_layer(self, input_shape, pool_size, inputs=None, name=None, **kwargs):
        kwargs.setdefault("label", name)
        layer = PoolLayer(input_shape, pool_size, **kwargs)
        return self.add_layer(layer, inputs=inputs, name=name)

    def compute(self, inputs, output):
        raise NotImplementedError()


class SequentialNetwork(DeepNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layers = []
        self.layers_by_name = {}

    @property
    def input(self):
        return None if len(self.layers) == 0 else self.layers[0].input

    @property
    def output(self):
        return None if len(self.layers) == 0 else self.layers[-1].output

    @with_self
    def add_layer(self, layer, inputs=None, name=None):
        assert isinstance(layer, Layer)
        assert layer not in self.layers

        if inputs is None:
            inputs = [self.layers[-1]] if self.output is not None else []

        assert (
            len(inputs) == 0
            if self.output is None
            else (len(inputs) == 1 and inputs[0] is self.layers[-1])
        )

        for i in inputs:
            nengo.Connection(i.output, layer.input, synapse=None, **layer.pre_args)
        self.layers.append(layer)

        if name is not None:
            assert name not in self.layers_by_name
            self.layers_by_name[name] = layer

        return layer

    def layers_to(self, end):
        if end is None:
            return self.layers

        assert end in self.layers
        return self.layers[: self.layers.index(end) + 1]

    def compute(self, inputs, output=None):
        y = inputs
        for layer in self.layers_to(output):
            y = layer.compute(y)

        return y


class Layer(nengo.Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def _compute_input(self, x):
        x = np.asarray(x)
        if x.ndim == 0:
            raise ValueError("'x' must be at least 1D")
        if x.ndim == 1:
            assert x.shape[0] == self.size_in
            return x.reshape((1, -1))

        assert x.ndim == 2 and x.shape[1] == self.size_in
        return x


class NodeLayer(Layer):
    def __init__(self, output=None, size_in=None, **kwargs):
        super().__init__(**kwargs)

        with self:
            self.node = nengo.Node(output=output, size_in=size_in)

        if self.label is not None:
            self.node.label = "%s_node" % self.label

    @property
    def input(self):
        return self.node

    @property
    def output(self):
        return self.node

    def compute(self, x):
        raise NotImplementedError()


class NeuronLayer(Layer):
    def __init__(
        self,
        n,
        neuron_type=Default,
        synapse=Default,
        gain=1.0,
        bias=0.0,
        amplitude=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        with self:
            self.ensemble = nengo.Ensemble(n, 1, neuron_type=neuron_type)
            self.ensemble.gain = _vectorize(gain, n, "gain")
            self.ensemble.bias = _vectorize(bias, n, "bias")

            self._output = nengo.Node(size_in=n)
            self.connection = nengo.Connection(
                self.ensemble.neurons, self.output, transform=amplitude, synapse=synapse
            )

        if self.label is not None:
            self.ensemble.label = "%s_ensemble" % self.label
            self.output.label = "%s_output" % self.label

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


class DataLayer(NodeLayer):
    def __init__(self, size, **kwargs):
        super().__init__(size_in=size, **kwargs)

    def compute(self, x):
        return x.reshape((x.shape[0], self.size_out))


class SoftmaxLayer(NodeLayer):
    def __init__(self, size, **kwargs):
        super().__init__(output=lambda t, x: softmax(x), size_in=size, **kwargs)

    def compute(self, x):
        x = self._compute_input(x)
        return softmax(x, axis=-1)


class DropoutLayer(NodeLayer):
    def __init__(self, size, keep, **kwargs):
        super().__init__(size_in=size, **kwargs)
        self.pre_args["transform"] = keep

    @property
    def keep(self):
        return self.pre_args["transform"]

    def compute(self, x):
        x = self._compute_input(x)
        return self.keep * x


class FullLayer(NodeLayer):
    def __init__(self, weights, biases, **kwargs):
        assert weights.ndim == 2
        assert biases.size == weights.shape[0]
        super().__init__(size_in=weights.shape[0], **kwargs)

        self.weights = np.array(weights)  # copy
        self.biases = np.array(biases)  # copy

        with self:
            self.bias = nengo.Node(output=biases)
            nengo.Connection(self.bias, self.node, synapse=None)

        self.pre_args["transform"] = weights

        if self.label is not None:
            self.bias.label = "%s_bias" % self.label

    def compute(self, x):
        x = self._compute_input(x)
        return np.dot(x, self.weights.T) + self.biases


class ProcessLayer(NodeLayer):
    def __init__(self, process, **kwargs):
        assert isinstance(process, nengo.Process)
        super().__init__(output=process, **kwargs)
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
                self.size_in, self.size_out, dt=1.0, rng=None, state={}
            )
        return self._step(0, x).reshape((n, -1))


class LocalLayer(ProcessLayer):
    def __init__(self, input_shape, filters, biases, strides=1, padding=0, **kwargs):
        assert filters.ndim == 6
        filters = np.array(filters)  # copy
        biases = np.array(biases)  # copy
        p = Conv2d(input_shape, filters, biases, strides=strides, padding=padding)
        super().__init__(p, **kwargs)


class ConvLayer(ProcessLayer):
    def __init__(
        self,
        input_shape,
        filters,
        biases,
        strides=1,
        padding=0,
        border="ceil",
        **kwargs,
    ):
        assert filters.ndim == 4
        filters = np.array(filters)  # copy
        biases = np.array(biases)  # copy
        p = Conv2d(
            input_shape,
            filters,
            biases,
            strides=strides,
            padding=padding,
            border=border,
        )
        super().__init__(p, **kwargs)


class PoolLayer(ProcessLayer):
    def __init__(
        self, input_shape, pool_size, strides=None, kind="avg", mode="full", **kwargs
    ):
        p = Pool2d(input_shape, pool_size, strides=strides, kind=kind, mode=mode)
        super().__init__(p, **kwargs)


def _vectorize(value, n, name):
    value = np.asarray(value)
    if value.size == 1:
        return value * np.ones(n)
    if value.size == n:
        return value
    raise ValueError("%r must be length 1 or %d (got %d)" % (name, n, value.size))
