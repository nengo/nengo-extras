import numpy as np
import pytest


@pytest.mark.xfail
def test_softlif_layer(plt):
    pytest.importorskip('keras')
    import keras.models
    import nengo_extras.keras

    model = keras.models.Sequential()
    model.add(nengo_extras.keras.SoftLIF(input_shape=(1,)))

    x = np.linspace(-10, 30, 256).reshape(-1, 1)
    y = model.predict(x)
    y0 = nengo_extras.SoftLIFRate().rates(x, 1., 1.)

    plt.plot(x, y)
    plt.plot(x, y0, 'k--')

    assert np.allclose(y, y0, atol=1e-5, rtol=1e-5)
