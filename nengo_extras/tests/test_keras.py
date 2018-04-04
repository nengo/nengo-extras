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
    df = K.function([x], K.gradients(y, [x]))

    x = np.linspace(-10, 30, 1024).reshape(-1, 1)
    dy = df([x])[0]
    dy0 = nengo_extras.SoftLIFRate(**params).derivative(x, 1., 1.)

    plt.plot(x, dy)
    plt.plot(x, dy0, 'k--')

    assert np.isfinite(dy).all()
    assert np.allclose(dy, dy0, atol=1e-2, rtol=1e-3)
