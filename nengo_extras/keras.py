from __future__ import absolute_import

import os

import nengo
import numpy as np

import nengo_extras.deepnetworks
from nengo_extras import reqs
from nengo_extras.neurons import SoftLIFRate

if reqs.HAS_TENSORFLOW:
    import tensorflow as tf

if reqs.HAS_KERAS:
    import keras
    from keras import backend as K
    from keras.layers import Convolution2D, Layer, LocallyConnected2D
else:

    class Layer:
        pass


class SoftLIF(Layer):
    def __init__(
        self,
        sigma=1.0,
        amplitude=1.0,
        tau_rc=0.02,
        tau_ref=0.002,
        noise_model="none",
        tau_s=0.005,
        **kwargs,
    ):
        if not reqs.HAS_KERAS:
            raise ImportError("`SoftLIF` requires `keras`")

        self.supports_masking = True
        self.sigma = sigma
        self.amplitude = amplitude
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

        if noise_model not in ("none", "alpharc"):
            raise ValueError("Unrecognized noise model")
        self.noise_model = noise_model

        self.tau_s = tau_s
        if abs(self.tau_rc - self.tau_s) < 1e-4:
            raise ValueError("tau_rc and tau_s must be different")

        super().__init__(**kwargs)

    def call(self, x, mask=None):
        """Compute the SoftLIF nonlinearity."""

        if not reqs.HAS_TENSORFLOW:
            raise ImportError('Keras backend is "tensorflow" but it cannot be imported')

        assert K.backend() == "tensorflow"

        # compute non-noisy output
        xs = x / self.sigma
        x_valid = xs > -20
        xs_safe = tf.where(x_valid, xs, K.zeros_like(xs))
        j = K.softplus(xs_safe) * self.sigma
        p = self.tau_ref + self.tau_rc * tf.where(
            x_valid, tf.math.log1p(1 / j), -xs - np.log(self.sigma)
        )
        r = self.amplitude / p

        if self.noise_model == "none":
            return r

        assert self.noise_model == "alpharc"
        # compute noisy output for forward pass
        d = self.tau_rc - self.tau_s
        u01 = K.random_uniform(K.shape(p))
        t = u01 * p
        q_rc = K.exp(-t / self.tau_rc)
        q_s = K.exp(-t / self.tau_s)
        r_rc1 = -tf.math.expm1(-p / self.tau_rc)  # 1 - exp(-p/tau_rc)
        r_s1 = -tf.math.expm1(-p / self.tau_s)  # 1 - exp(-p/tau_s)

        pt = tf.where(p < 100 * self.tau_s, (p - t) * (1 - r_s1), K.zeros_like(p))
        qt = tf.where(t < 100 * self.tau_s, q_s * (t + pt), K.zeros_like(t))
        rt = qt / (self.tau_s * d * r_s1 ** 2)
        rn = self.tau_rc * (q_rc / (d * d * r_rc1) - q_s / (d * d * r_s1)) - rt

        # r + stop_gradient(rn - r) = rn on forward pass, r on backwards
        return r + K.stop_gradient(self.amplitude * rn - r)

    def get_config(self):
        """Return a config dict to reproduce this SoftLIF."""
        config = {
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "tau_rc": self.tau_rc,
            "tau_ref": self.tau_ref,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


_custom_objects = {"SoftLIF": SoftLIF} if reqs.HAS_KERAS else {}


def load_model_pair(filepath, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}

    json_path = filepath + ".json"
    h5_path = filepath + ".h5"

    combined_customs = dict(_custom_objects)
    combined_customs.update(custom_objects)

    with open(json_path, "r", encoding="utf-8") as f:
        model = keras.models.model_from_json(f.read(), custom_objects=combined_customs)

    model.load_weights(h5_path)
    return model


def save_model_pair(model, filepath, overwrite=False):
    json_path = filepath + ".json"
    h5_path = filepath + ".h5"

    if not overwrite and os.path.exists(json_path):
        raise ValueError("Path already exists: %r" % filepath)

    json_string = model.to_json()
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)

    model.save_weights(h5_path, overwrite=overwrite)


def kmodel_compute_shapes(kmodel, input_shape):
    assert isinstance(kmodel, keras.models.Sequential)

    shapes = [input_shape]
    for layer in kmodel.layers:
        s = layer.compute_output_shape(shapes[-1])
        shapes.append(s)

    return shapes


class SequentialNetwork(nengo_extras.deepnetworks.SequentialNetwork):
    def __init__(self, model, synapse=None, lif_type="lif", **kwargs):
        super().__init__(**kwargs)

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

        raise NotImplementedError("Cannot build layer type %r" % type(layer).__name__)

    def _add_dense_layer(self, layer):
        weights, biases = layer.get_weights()
        return self.add_full_layer(weights.T, biases, name=layer.name)

    def _add_conv2d_layer(self, layer):
        if not reqs.HAS_KERAS:
            raise ImportError("`_add_conv2d_layer` requires `keras`")

        shape_in = layer.input_shape[1:]
        filters, biases = layer.get_weights()
        strides = layer.strides

        assert layer.data_format == "channels_first"
        nc, _, _ = shape_in
        filters = np.transpose(filters, (3, 2, 0, 1))
        assert K.backend() == "tensorflow"
        filters = filters.copy()  # to make contiguous

        _, nc2, si, sj = filters.shape
        assert nc == nc2, "Filter channels must match input channels"

        if layer.padding == "valid":
            padding = (0, 0)
        elif layer.padding == "same":
            padding = ((si - 1) / 2, (sj - 1) / 2)
        else:
            raise ValueError("Unrecognized padding %r" % layer.padding)

        conv = self.add_conv_layer(
            shape_in,
            filters,
            biases,
            strides=strides,
            padding=padding,
            border="floor",
            name=layer.name,
        )
        assert conv.size_out == np.prod(layer.output_shape[1:])
        return conv

    def _add_pool2d_layer(self, layer, kind=None):
        shape_in = layer.input_shape[1:]
        pool_size = layer.pool_size
        strides = layer.strides
        return self.add_pool_layer(
            shape_in,
            pool_size,
            strides=strides,
            kind=kind,
            mode="valid",
            name=layer.name,
        )

    def _add_avgpool2d_layer(self, layer):
        return self._add_pool2d_layer(layer, kind="avg")

    def _add_maxpool2d_layer(self, layer):
        return self._add_pool2d_layer(layer, kind="max")

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
            raise ValueError("Unrecognized activation type %r" % layer.activation)

        n = np.prod(layer.input_shape[1:])
        return self.add_neuron_layer(
            n,
            neuron_type=neuron_type,
            synapse=self.synapse,
            gain=1,
            bias=0,
            name=layer.name,
        )

    def _add_softlif_layer(self, layer):

        taus = dict(tau_rc=layer.tau_rc, tau_ref=layer.tau_ref)
        lif_type = self.lif_type.lower()
        if lif_type == "lif":
            neuron_type = nengo.LIF(**taus)
        elif lif_type == "lifrate":
            neuron_type = nengo.LIFRate(**taus)
        elif lif_type == "softlifrate":
            neuron_type = SoftLIFRate(sigma=layer.sigma, **taus)
        else:
            raise KeyError("Unrecognized LIF type %r" % self.lif_type)

        n = np.prod(layer.input_shape[1:])
        return self.add_neuron_layer(
            n,
            neuron_type=neuron_type,
            synapse=self.synapse,
            gain=1,
            bias=1,
            amplitude=layer.amplitude,
            name=layer.name,
        )

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


def LSUVinit(kmodel, X, tol=0.1, t_max=50):  # noqa: C901
    """Layer-sequential unit-variance initialization.

    References
    ----------
    .. [1] Mishkin, D., & Matas, J. (2016). All you need is a good init.
       In ICLR 2016 (pp. 1-13).
    """
    if not reqs.HAS_KERAS:
        raise ImportError("`LSUVinit` requires `keras`")

    # f = K.function([kmodel.layers[0].input, K.learning_phase()],
    #                [klayer.output])
    # --- orthogonalize weights
    def orthogonalize(X):
        assert X.ndim == 2
        U, _, V = np.linalg.svd(X, full_matrices=False)
        return np.dot(U, V)

    for layer in kmodel.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        W, b = weights
        if isinstance(layer, Convolution2D):
            Wv = W.reshape((W.shape[0], -1))
        elif isinstance(layer, LocallyConnected2D):
            Wv = W.reshape((-1, W.shape[-1]))
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

        for _ in range(t_max):
            Y = f([X, learning_phase])[0]
            Ystd = Y.std()
            print(Ystd)
            if abs(Ystd - 1) < tol:
                break

            W /= Ystd
            layer.set_weights((W, b))
        else:
            print(
                "Layer %r did not converge after %d iterations (Ystd=%0.3e)"
                % (layer.name, t_max, Ystd)
            )
