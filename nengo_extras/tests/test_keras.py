import nengo
import numpy as np
import pytest


def test_softlif_layer(plt):
    pytest.importorskip('keras')
    import keras.models
    import nengo_extras.keras

    params = dict(sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.001)

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(input_shape=(1,), **params))

    x = np.linspace(-10, 30, 256).reshape(-1, 1)
    y = model.predict(x)
    y0 = nengo_extras.SoftLIFRate(**params).rates(x, 1., 1.)

    plt.plot(x, y)
    plt.plot(x, y0, 'k--')

    assert np.isfinite(y).all()
    assert np.allclose(y, y0, atol=1e-3, rtol=1e-3)


def test_softlif_derivative(plt):
    pytest.importorskip('keras')
    import keras.models
    import nengo_extras.keras
    from keras import backend as K

    params = dict(sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.001)

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(input_shape=(1,), **params))

    x = model.layers[0].input
    y = model.layers[0].output
    df = K.function([x], K.gradients(K.sum(y), [x]))

    x = np.linspace(-10, 30, 1024).reshape(-1, 1)
    dy = df([x])[0]
    dy0 = nengo_extras.SoftLIFRate(**params).derivative(x, 1., 1.)

    plt.plot(x, dy)
    plt.plot(x, dy0, 'k--')

    assert np.isfinite(dy).all()
    assert np.allclose(dy, dy0, atol=1e-2, rtol=1e-3)


def test_conv2d_layer(Simulator, seed, rng, plt):
    pytest.importorskip('keras')
    import keras
    import nengo_extras.keras as nekeras
    np.random.seed(seed)  # for Keras weights

    data_format = 'channels_first'
    img_shape = (1, 28, 28)
    presentation_time = 0.01

    nb_filters = 4
    nb_conv = 3

    kmodel = keras.models.Sequential()
    kmodel.add(keras.layers.Conv2D(
        nb_filters, (nb_conv, nb_conv), padding='valid', strides=(2, 2),
        input_shape=img_shape, data_format=data_format))

    X = rng.uniform(-1, 1, size=(1,) + img_shape)
    output_shape = nekeras.kmodel_compute_shapes(kmodel, X.shape)[-1][1:]

    with nengo.Network() as model:
        u = nengo.Node(nengo.processes.PresentInput(X, presentation_time))
        knet = nekeras.SequentialNetwork(
            kmodel, synapse=None, lif_type='lifrate')
        nengo.Connection(u, knet.input, synapse=None)
        output_p = nengo.Probe(knet.output)

    with Simulator(model) as sim:
        sim.run(presentation_time)

    y0 = kmodel.predict(X)[0]
    y1 = sim.data[output_p][-1].reshape(output_shape)
    assert np.allclose(y1, y0, atol=1e-5, rtol=1e-5)
