from __future__ import absolute_import
import os

import keras
import nengo
import numpy as np

import nengo_extras.deepnetworks


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


def load_model_pair(filepath, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}

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


class SequentialNetwork(nengo_extras.deepnetworks.SequentialNetwork):

    def __init__(self, model, synapse=None, lif_type='lif', **kwargs):
        super(SequentialNetwork, self).__init__(**kwargs)

        assert isinstance(model, keras.models.Sequential)
        self.model = model
        self.synapse = synapse
        self.lif_type = lif_type

        # -- build model
        self.add_data_layer(np.prod(model.input_shape[1:]))
        for layer in model.layers:
            self._add_layer(layer)

    def _add_layer(self, layer):
        assert layer.input_mask is None
        assert layer.input_shape[0] is None

        layer_adder = {
            keras.layers.Activation: self._add_activation_layer,
            keras.layers.Dense: self._add_dense_layer,
            keras.layers.Dropout: self._add_dropout_layer,
            keras.layers.Flatten: self._add_flatten_layer,
            keras.layers.Convolution2D: self._add_conv2d_layer,
            keras.layers.AveragePooling2D: self._add_avgpool2d_layer,
            keras.layers.MaxPooling2D: self._add_maxpool2d_layer,
            keras.layers.noise.GaussianNoise: self._add_gaussian_noise_layer,
            SoftLIF: self._add_softlif_layer,
        }

        for cls in type(layer).__mro__:
            if cls in layer_adder:
                return layer_adder[cls](layer)

        raise NotImplementedError("Cannot build layer type %r" %
                                  type(layer).__name__)

    def _add_dense_layer(self, layer):
        weights, biases = layer.get_weights()
        return self.add_full_layer(weights.T, biases, name=layer.name)

    def _add_conv2d_layer(self, layer):
        shape_in = layer.input_shape[1:]
        filters, biases = layer.get_weights()
        filters = filters[..., ::-1, ::-1]  # flip
        strides = layer.subsample

        nf, nc, ni, nj = filters.shape
        if layer.border_mode == 'valid':
            padding = (0, 0)
        elif layer.border_mode == 'same':
            padding = ((ni - 1) / 2, (nj - 1) / 2)
        else:
            raise ValueError("Unrecognized border mode %r" % layer.border_mode)

        return self.add_conv_layer(shape_in, filters, biases, strides=strides,
                                   padding=padding, name=layer.name)

    def _add_pool2d_layer(self, layer, kind=None):
        shape_in = layer.input_shape[1:]
        pool_size = layer.pool_size
        strides = layer.strides
        return self.add_pool_layer(shape_in, pool_size, strides=strides,
                                   kind=kind, mode='valid', name=layer.name)

    def _add_avgpool2d_layer(self, layer):
        return self._add_pool2d_layer(layer, kind='avg')

    def _add_maxpool2d_layer(self, layer):
        return self._add_pool2d_layer(layer, kind='max')

    def _add_activation_layer(self, layer):
        if layer.activation is keras.activations.softmax:
            return self._add_softmax_layer(layer)

        # add normal activation layer
        activation_map = {
            keras.activations.relu: nengo.neurons.RectifiedLinear(),
            keras.activations.sigmoid: nengo.neurons.Sigmoid(),
            }
        neuron_type = activation_map.get(layer.activation, None)
        if neuron_type is None:
            raise ValueError("Unrecognized activation type %r"
                             % layer.activation)

        n = np.prod(layer.input_shape[1:])
        return self.add_neuron_layer(
            n, neuron_type=neuron_type, synapse=self.synapse,
            gain=1, bias=0, name=layer.name)

    def _add_softlif_layer(self, layer):
        from .neurons import SoftLIFRate

        taus = dict(tau_rc=layer.tau_rc, tau_ref=layer.tau_ref)
        lif_type = self.lif_type.lower()
        if lif_type == 'lif':
            neuron_type = nengo.LIF(**taus)
        elif lif_type == 'lifrate':
            neuron_type = nengo.LIFRate(**taus)
        elif lif_type == 'softlifrate':
            neuron_type = SoftLIFRate(sigma=layer.sigma, **taus)
        else:
            raise KeyError("Unrecognized LIF type %r" % self.lif_type)

        n = np.prod(layer.input_shape[1:])
        return self.add_neuron_layer(
            n, neuron_type=neuron_type, synapse=self.synapse,
            gain=1, bias=1, amplitude=layer.amplitude, name=layer.name)

    def _add_softmax_layer(self, layer):
        return None  # non-neural, we can do without it
        # return self.add_softmax_layer(
        #     np.prod(layer.input_shape[1:]), name=layer.name)

    def _add_dropout_layer(self, layer):
        return None  # keras scales by dropout rate, so we don't have to

    def _add_flatten_layer(self, layer):
        return None  # no computation, just reshaping, so ignore

    def _add_gaussian_noise_layer(self, layer):
        return None  # no noise during testing


def LSUVinit(kmodel, X, tol=0.1, t_max=50):
    """Layer-sequential unit-variance initialization.

    References
    ----------
    .. [1] Mishkin, D., & Matas, J. (2016). All you need is a good init.
       In ICLR 2016 (pp. 1-13).
    """
    from keras.layers import Convolution2D, LocallyConnected2D
    import keras.backend as K
    # f = K.function([kmodel.layers[0].input, K.learning_phase()],
    #                [klayer.output])

    # --- orthogonalize weights
    def orthogonalize(X):
        assert X.ndim == 2
        U, s, V = np.linalg.svd(X, full_matrices=False)
        return np.dot(U, V)

    for layer in kmodel.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        W, b = weights
        if isinstance(layer, Convolution2D):
            Wv = W.reshape(W.shape[0], -1)
        elif isinstance(layer, LocallyConnected2D):
            Wv = W.reshape(-1, W.shape[-1])
        else:
            assert W.ndim == 2
            Wv = W

        Wv[:] = orthogonalize(Wv)
        layer.set_weights((W, b))

    # --- adjust variances
    s_input = kmodel.layers[0].input
    for layer in kmodel.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        W, b = weights
        f = K.function([s_input, K.learning_phase()], [layer.output])
        learning_phase = 0  # 0 == testing, 1 == training

        for i in range(t_max):
            Y = f([X, learning_phase])[0]
            Ystd = Y.std()
            print(Ystd)
            if abs(Ystd - 1) < tol:
                break

            W /= Ystd
            layer.set_weights((W, b))
        else:
            print("Layer %r did not converge after %d iterations (Ystd=%0.3e)"
                  % (layer.name, t_max, Ystd))
