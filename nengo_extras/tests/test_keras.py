import numpy as np
import pytest


def test_softlif_layer(plt):
    pytest.importorskip('keras')
    import keras.models
    import nengo_extras.keras

    softlif_params = dict(
        sigma=1.0, amplitude=0.063, tau_rc=0.05, tau_ref=0.001)

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(
        input_shape=(1,), **softlif_params))

    x = np.linspace(-100, 100, 1001, dtype=np.float32).reshape(-1, 1)
    y = model.predict(x)
    y0 = nengo_extras.SoftLIFRate(**softlif_params).rates(x, 1., 1.)

    plt.plot(x, y)
    plt.plot(x, y0, 'k--')

    assert np.allclose(y, y0, atol=1e-5, rtol=1e-5)


def test_softlif_layer_noise(plt):
    pytest.importorskip('keras')
    import keras.backend as K
    import keras.models
    import nengo_extras.keras
    K.set_learning_phase(1)  # noise only in training

    softlif_params = dict(
        sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.002)

    nx = 101

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(
        input_shape=(nx,), noise=('gaussian', 10), **softlif_params))

    x = np.linspace(-10, 50, nx, dtype=np.float32)
    y0 = nengo_extras.SoftLIFRate(**softlif_params).rates(x, 1., 1.)

    n = 1000
    X = np.tile(x.reshape(1, -1), (n, 1))
    Y = model.predict(X)
    ymean = Y.mean(0)
    ystd = Y.std(0)

    plt.errorbar(x, ymean, ystd)
    plt.plot(x, y0, 'k--')
