import numpy as np
import pytest

import nengo
from nengo.utils.stdlib import Timer

from nengo_extras.routing import ARCLayer


def test_arc_1d_layers(rng, plt):

    y = np.array([-0.507776, -0.827827, -0.960391, -0.830145,
                  -0.457190, 0.047655, 0.524551, 0.826298, 0.874566, 0.687630,
                  0.368475, 0.058748, -0.121739, -0.124369, 0.010761, 0.176212,
                  0.247980, 0.143670, -0.137110])
    # x = np.linspace(-1, 1, 19)
    # y = np.sin(2*np.pi*x)

    layer_n = np.array([19, 13, 7])

    Alen = 9
    Apos = 3

    layer_mi = np.array([0] + list((layer_n[:-1] - layer_n[1:])/2))
    layer_m = np.cumsum(layer_mi)  # maximum shift for each layer

    sf = (np.clip(Alen, layer_n[1:], layer_n[:-1]) - 1) / (layer_n[1:] - 1.)
    theta = (abs(Apos) > layer_m) * (Apos - np.sign(Apos)*layer_m)

    with nengo.Network() as model:
        u = nengo.Node(y)

        layers = []
        for k, (nx, ny) in enumerate(zip(layer_n, layer_n[1:])):
            layer = ARCLayer((nx,), (ny,))
            layers.append(layer)
            nengo.Connection(nengo.Node(sf[k]), layer.scale)
            nengo.Connection(nengo.Node(theta[k]), layer.loc)

        nengo.Connection(u, layers[0].input)
        [nengo.Connection(a.output, b.input) for a, b in zip(layers, layers[1:])]

        in_p = nengo.Probe(u)
        layers_p = [nengo.Probe(layer.output) for layer in layers]

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    rows, cols = len(layers_p)+1, 1
    for k, p in enumerate([in_p] + layers_p):
        plt.subplot(rows, cols, k+1)
        plt.plot(sim.data[p][-1])


def test_arc_2d_layers(rng, plt):
    m, n = 30, 32

    image = rng.normal(size=(m, n))
    pi, pj = m/2, n/2
    fi, fj = m/4, n/4
    I = np.fft.fft2(np.pad(image, ((pi, pi), (pj, pj)), 'constant'))
    I[fi:, :] = I[:, fj:] = 0
    image[:] = np.array(np.fft.ifft2(I).real[pi:-pi, pj:-pj])
    image[:] = 2*(image - image.min()) / (image.max() - image.min()) - 1

    layer_rfs = np.array([1, 7, 7, 7])[:, None]
    layer_n = np.array([m, n]) - np.cumsum(layer_rfs - 1, axis=0)

    # Alen = np.array([m, n])
    # Apos = np.array([0, 0])
    Alen = np.array([m-2, n-2])
    Apos = np.array([-8, -8])

    layer_mi = np.pad((layer_n[:-1] - layer_n[1:])/2, ((1, 0), (0, 0)), 'constant')
    layer_m = np.cumsum(layer_mi, axis=0)  # maximum shift for each layer

    sf = (np.clip(Alen, layer_n[1:], layer_n[:-1]) - 1) / (layer_n[1:] - 1.)
    theta = (abs(Apos) > layer_m) * (Apos - np.sign(Apos)*layer_m)

    with nengo.Network() as model:
        u = nengo.Node(image.ravel(), label='input')

        layers = []
        for k, (sx, sy) in enumerate(zip(layer_n, layer_n[1:])):
            layer = ARCLayer(sx, sy, axes=(-2, -1), label='arc_%d' % k)
            layers.append(layer)
            nengo.Connection(nengo.Node(np.array(sf[k])), layer.scale)
            nengo.Connection(nengo.Node(np.array(theta[k])), layer.loc)

        nengo.Connection(u, layers[0].input)
        [nengo.Connection(a.output, b.input) for a, b in zip(layers, layers[1:])]

        in_p = nengo.Probe(u)
        layers_p = [nengo.Probe(layer.output) for layer in layers]

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    assert len(layers_p)+1 == 4
    rows, cols = 2, 2
    for k, p in enumerate([in_p] + layers_p):
        plt.subplot(rows, cols, k+1)
        x = sim.data[p][-1].reshape(layer_n[k])
        plt.imshow(x, cmap='gray', interpolation='none')
        plt.title("min=%0.3f, max=%0.3f" % (x.min(), x.max()))
