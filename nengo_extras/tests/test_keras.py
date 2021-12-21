import nengo
import numpy as np
import pytest

import nengo_extras.keras
from nengo_extras.keras import SequentialNetwork, kmodel_compute_shapes


@pytest.mark.parametrize("noise_model", ("none", "alpharc"))
def test_softlif_layer(noise_model, plt, allclose):
    keras = pytest.importorskip("keras")

    nx = 256
    ni = 500

    params = dict(sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.001)
    layer_params = dict(params)
    layer_params["noise_model"] = noise_model
    if noise_model == "alpharc":
        layer_params["tau_s"] = 0.001

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(input_shape=(nx,), **layer_params))

    x = np.linspace(-10, 30, nx)
    y = model.predict(np.ones((ni, 1)) * x)
    y0 = nengo_extras.neurons.SoftLIFRate(**params).rates(x, 1.0, 1.0).ravel()

    plt.plot(x, y.mean(axis=0), "b")
    if noise_model != "none":
        plt.plot(x, np.percentile(y, 5, axis=0), "b:")
        plt.plot(x, np.percentile(y, 95, axis=0), "b:")
    plt.plot(x, y0, "k--")

    assert np.isfinite(y).all()
    if noise_model == "none":
        assert allclose(y, y0, atol=1e-3, rtol=1e-3)
    else:
        assert allclose(y.mean(axis=0), y0, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("noise_model", ("none", "alpharc"))
def test_softlif_derivative(noise_model, plt, allclose):
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    K = keras.backend

    # TODO: make this work with eager execution
    tf.compat.v1.disable_eager_execution()

    params = dict(sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.001)
    layer_params = dict(params)
    layer_params["noise_model"] = noise_model

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(input_shape=(1,), **layer_params))

    x = model.layers[0].input
    y = model.layers[0].output
    df = K.function([x], K.gradients(K.sum(y), [x]))

    x = np.linspace(-10, 30, 1024).reshape((-1, 1))
    dy = df([x])[0]
    dy0 = nengo_extras.neurons.SoftLIFRate(**params).derivative(x, 1.0, 1.0)

    plt.plot(x, dy)
    plt.plot(x, dy0, "k--")

    assert np.isfinite(dy).all()
    assert allclose(dy, dy0, atol=1e-2, rtol=1e-3)


def test_conv2d_layer(Simulator, seed, rng):
    keras = pytest.importorskip("keras")

    np.random.seed(seed)  # for Keras weights

    data_format = "channels_first"
    img_shape = (1, 28, 28)
    presentation_time = 0.01

    nb_filters = 4
    nb_conv = 3

    kmodel = keras.models.Sequential()
    kmodel.add(
        keras.layers.Conv2D(
            nb_filters,
            (nb_conv, nb_conv),
            padding="valid",
            strides=(2, 2),
            input_shape=img_shape,
            data_format=data_format,
        )
    )

    X = rng.uniform(-1, 1, size=(1,) + img_shape)
    output_shape = kmodel_compute_shapes(kmodel, X.shape)[-1][1:]

    with nengo.Network() as model:
        u = nengo.Node(nengo.processes.PresentInput(X, presentation_time))
        knet = SequentialNetwork(kmodel, synapse=None, lif_type="lifrate")
        nengo.Connection(u, knet.input, synapse=None)
        output_p = nengo.Probe(knet.output)

    with Simulator(model) as sim:
        sim.run(presentation_time)

    y0 = kmodel.predict(X)[0]
    y1 = sim.data[output_p][-1].reshape(output_shape)
    assert np.allclose(y1, y0, atol=1e-5, rtol=1e-5)
