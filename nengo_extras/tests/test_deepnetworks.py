import numpy as np
import pytest


from nengo_extras.deepnetworks import ConvLayer, LocalLayer, NeuronLayer


@pytest.mark.parametrize('local', (False, True))
def test_convlayer_theano(local, rng):
    pytest.importorskip('theano')
    import theano
    import theano.tensor as tt

    n = 2
    nf = 3
    nc = 4
    nxi, nxj = 6, 8
    si, sj = 3, 5
    pi, pj = 2, 1
    sti, stj = 1, 1
    nyi = 1 + max(int(np.ceil(float(2*pi + nxi - si) / sti)), 0)
    nyj = 1 + max(int(np.ceil(float(2*pj + nxj - sj) / stj)), 0)

    if local:
        filters = rng.uniform(-1, 1, size=(nf, nyi, nyj, nc, si, sj))
    else:
        filters = rng.uniform(-1, 1, size=(nf, nc, si, sj))

    biases = rng.uniform(-1, 1, size=(nf,))
    inputs = rng.uniform(-1, 1, size=(n, nc, nxi, nxj))

    if local:
        layer = LocalLayer((nc, nxi, nxj), filters, biases,
                           strides=(sti, stj), padding=(pi, pj))
    else:
        layer = ConvLayer((nc, nxi, nxj), filters, biases,
                          strides=(sti, stj), padding=(pi, pj))

    y0 = layer.compute(inputs.reshape(n, -1))

    sx = tt.matrix()
    f = theano.function([sx], layer.theano(sx), allow_input_downcast=True)
    y1 = f(inputs.reshape(n, -1))

    assert y0.shape == y1.shape == (n, nf*nyi*nyj)
    assert np.allclose(y1, y0, atol=1e-7)


@pytest.mark.xfail
def test_neuronlayer_softlif_theano():
    pytest.importorskip('theano')
    pytest.importorskip('keras')
    import theano
    import theano.tensor as tt
    from nengo_extras.keras import SoftLIF as SoftLIFLayer
    from nengo_extras.neurons import SoftLIFRate

    klayer = SoftLIFLayer()
    sx = tt.matrix()
    f0 = theano.function([sx], klayer.call(sx), allow_input_downcast=True)

    neuron_type = SoftLIFRate()
    layer = NeuronLayer(1, neuron_type=neuron_type, bias=1)
    sx = tt.matrix()
    f1 = theano.function([sx], layer.theano(sx), allow_input_downcast=True)

    x = np.linspace(-10, 10, 201).reshape(1, -1)
    y0 = f0(x)
    y1 = f1(x)

    assert y0.shape == y1.shape
    assert np.allclose(y0, y1, atol=1e-7)
