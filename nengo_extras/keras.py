from __future__ import absolute_import
import os

import keras
import nengo
import numpy as np


class SoftLIF(keras.layers.Layer):
    def __init__(self, sigma=1., amplitude=1., tau_rc=0.02, tau_ref=0.002,
                 **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.amplitude = amplitude
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        super(SoftLIF, self).__init__(**kwargs)

    def call(self, x, mask=None):
        from keras import backend as K
        j = K.softplus(x / self.sigma) * self.sigma
        r = self.amplitude / (self.tau_ref + self.tau_rc*K.log(1 + 1/j))
        return K.switch(j > 0, r, 0)

    def get_config(self):
        config = {'sigma': self.sigma, 'amplitude': self.amplitude,
                  'tau_rc': self.tau_rc, 'tau_ref': self.tau_ref}
        base_config = super(SoftLIF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


_custom_objects = {
    'SoftLIF': SoftLIF
}


def load_model_pair(filepath, custom_objects={}):
    json_path = filepath + '.json'
    h5_path = filepath + '.h5'

    combined_customs = dict(_custom_objects)
    combined_customs.update(custom_objects)

    with open(json_path, 'r') as f:
        model = keras.models.model_from_json(
            f.read(), custom_objects=combined_customs)

    model.load_weights(h5_path)
    return model


def save_model_pair(model, filepath, overwrite=False):
    json_path = filepath + '.json'
    h5_path = filepath + '.h5'

    if not overwrite and os.path.exists(json_path):
        raise ValueError("Path already exists: %r" % filepath)

    json_string = model.to_json()
    with open(json_path, 'w') as f:
        f.write(json_string)

    model.save_weights(h5_path, overwrite=overwrite)


class SequentialNetwork(nengo.Network):

    def __init__(self, model, synapse=None, spiking=True, **kwargs):
        super(SequentialNetwork, self).__init__(**kwargs)
        assert isinstance(model, keras.models.Sequential)
        self.model = model
        self.synapse = synapse
        self.spiking = spiking

        self.layers = []
        with self:
            self.input = nengo.Node(size_in=np.prod(model.input_shape[1:]),
                                    label='%s_input' % model.name)
            self.layers.append(self.input)
            for layer in model.layers:
                self.layers.append(self.add_layer(self.layers[-1], layer))

            self.output = self.layers[-1]

    def add_layer(self, pre, layer):
        assert layer.input_mask is None
        assert layer.input_shape[0] is None

        layer_adder = {
            keras.layers.convolutional.Convolution2D: self._add_conv2d_layer,
            keras.layers.convolutional.AveragePooling2D:
            self._add_avgpool2d_layer,
            keras.layers.convolutional.MaxPooling2D: self._add_maxpool2d_layer,
            keras.layers.core.Activation: self._add_activation_layer,
            keras.layers.core.Dense: self._add_dense_layer,
            keras.layers.core.Dropout: self._add_dropout_layer,
            keras.layers.core.Flatten: self._add_flatten_layer,
            SoftLIF: self._add_softlif_layer,
        }

        for cls in type(layer).__mro__:
            if cls in layer_adder:
                return layer_adder[cls](pre, layer)

        raise NotImplementedError("Cannot build layer type %r" %
                                  type(layer).__name__)

    def _add_dense_layer(self, pre, layer):
        weights, biases = layer.get_weights()
        node = nengo.Node(size_in=layer.output_dim, label=layer.name)
        b = nengo.Node(biases, label='%s_biases' % layer.name)
        nengo.Connection(pre, node, transform=weights.T, synapse=None)
        nengo.Connection(b, node, synapse=None)
        return node

    def _add_conv2d_layer(self, pre, layer):
        from .convnet import Conv2d

        shape_in = layer.input_shape[1:]
        filters, biases = layer.get_weights()
        strides = layer.subsample

        nf, nc, ni, nj = filters.shape
        if layer.border_mode == 'valid':
            padding = (0, 0)
        elif layer.border_mode == 'same':
            padding = ((ni - 1) / 2, (nj - 1) / 2)
        else:
            raise ValueError("Unrecognized border mode %r" % layer.border_mode)

        conv2d = Conv2d(
            shape_in, filters, biases=biases, strides=strides, padding=padding)
        assert conv2d.shape_out == layer.output_shape[1:]
        node = nengo.Node(conv2d, label=layer.name)
        nengo.Connection(pre, node, synapse=None)
        return node

    def _add_pool2d_layer(self, pre, layer, kind=None):
        from .convnet import Pool2d
        shape_in = layer.input_shape[1:]
        pool_size = layer.pool_size
        strides = layer.strides
        pool2d = Pool2d(shape_in, pool_size, strides=strides, kind=kind,
                        mode='valid')
        assert pool2d.shape_out == layer.output_shape[1:]
        node = nengo.Node(pool2d, label=layer.name)
        nengo.Connection(pre, node, synapse=None)
        return node

    def _add_avgpool2d_layer(self, pre, layer):
        return self._add_pool2d_layer(pre, layer, kind='avg')

    def _add_maxpool2d_layer(self, pre, layer):
        return self._add_pool2d_layer(pre, layer, kind='max')

    def _add_softmax_layer(self, pre, layer):
        from .convnet import softmax
        node = nengo.Node(lambda t, x: softmax(x),
                          size_in=np.prod(layer.input_shape[1:]),
                          label=layer.name)
        nengo.Connection(pre, node, synapse=None)
        return node

    def _add_activation_layer(self, pre, layer):
        if layer.activation is keras.activations.softmax:
            return self._add_softmax_layer(pre, layer)

        # add normal activation layer
        assert not self.spiking
        activation_map = {
            keras.activations.relu: nengo.neurons.RectifiedLinear(),
            keras.activations.sigmoid: nengo.neurons.Sigmoid(),
            }
        neuron_type = activation_map.get(layer.activation, None)
        if neuron_type is None:
            raise ValueError("Unrecognized activation type %r"
                             % layer.activation)

        n = np.prod(layer.input_shape[1:])
        e = nengo.Ensemble(n, 1, label='%s_neurons' % layer.name,
                           neuron_type=neuron_type)
        e.gain = np.ones(n)
        e.bias = np.zeros(n)
        node = nengo.Node(size_in=n, label=layer.name)
        nengo.Connection(pre, e.neurons, synapse=None)
        nengo.Connection(e.neurons, node, synapse=self.synapse)
        return node

    def _add_softlif_layer(self, pre, layer):
        from .neurons import SoftLIFRate
        taus = dict(tau_rc=layer.tau_rc, tau_ref=layer.tau_ref)
        neuron_type = (nengo.LIF(**taus) if self.spiking else
                       SoftLIFRate(sigma=layer.sigma, **taus))
        n = np.prod(layer.input_shape[1:])
        e = nengo.Ensemble(n, 1, label='%s_neurons' % layer.name,
                           neuron_type=neuron_type)
        e.gain = np.ones(n)
        e.bias = np.ones(n)
        node = nengo.Node(size_in=n, label=layer.name)
        nengo.Connection(pre, e.neurons, synapse=None)
        nengo.Connection(
            e.neurons, node, transform=layer.amplitude, synapse=self.synapse)
        return node

    def _add_dropout_layer(self, pre, layer):
        transform = 1. / (1 - layer.p)
        node = nengo.Node(size_in=np.prod(layer.output_shape[1:]),
                          label=layer.name)
        nengo.Connection(pre, node, transform=transform, synapse=None)
        return node

    def _add_flatten_layer(self, pre, layer):
        node = nengo.Node(size_in=np.prod(layer.output_shape[1:]),
                          label=layer.name)
        nengo.Connection(pre, node, synapse=None)
        return node
