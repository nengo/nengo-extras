import nengo
import numpy as np
import pytest
from nengo.utils.stdlib import Timer

from nengo_extras.convnet import Conv2d, Pool2d


@pytest.mark.parametrize("local", [False, True])
def test_conv2d(local, Simulator, rng):
    f = 4
    c = 2
    ni, nj = 30, 32
    si, sj = 5, 3

    # f = 64
    # c = 64
    # ni, nj = 32, 32
    # si, sj = 11, 11

    si2 = int((si - 1) / 2.0)
    sj2 = int((sj - 1) / 2.0)

    fshape = (f, ni, nj, c, si, sj) if local else (f, c, si, sj)
    filters = rng.uniform(-1, 1, size=fshape)
    biases = rng.uniform(-1, 1, size=f)
    image = rng.uniform(-1, 1, size=(c, ni, nj))

    model = nengo.Network()
    with model:
        u = nengo.Node(image.ravel())
        v = nengo.Node(Conv2d((c, ni, nj), filters, biases, padding=(si2, sj2)))
        nengo.Connection(u, v, synapse=None)
        vp = nengo.Probe(v)

    with Simulator(model) as sim:
        with Timer() as timer:
            sim.step()

    print("Conv2d(local=%s): %0.3e" % (local, timer.duration))

    # --- check result
    result = np.zeros((f, ni, nj))
    for i in range(ni):
        for j in range(nj):
            i0, i1 = i - si2, i + si2 + 1
            j0, j1 = j - sj2, j + sj2 + 1
            sli = slice(max(-i0, 0), min(ni + si - i1, si))
            slj = slice(max(-j0, 0), min(nj + sj - j1, sj))
            w = filters[:, i, j, :, sli, slj] if local else filters[:, :, sli, slj]
            xij = image[:, max(i0, 0) : min(i1, ni), max(j0, 0) : min(j1, nj)]
            result[:, i, j] += np.dot(xij.ravel(), w.reshape((f, -1)).T)

    result += biases.reshape((-1, 1, 1))

    y = sim.data[vp][-1].reshape((f, ni, nj))
    assert np.allclose(result, y, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("s, st", [(2, 2), (3, 1), (3, 3)])
def test_pool2d(s, st, Simulator, rng):
    nc = 3
    nxi, nxj = 30, 32
    image = rng.normal(size=(nc, nxi, nxj))

    # compute correct result
    nyi = 1 + int(np.ceil(float(nxi - s) / st))
    nyj = 1 + int(np.ceil(float(nxj - s) / st))
    nxi2, nxj2 = nyi * st, nyj * st
    result = np.zeros((nc, nyi, nyj))
    count = np.zeros((nyi, nyj))
    for i in range(s):
        for j in range(s):
            xij = image[:, i : min(nxi2 + i, nxi) : st, j : min(nxj2 + j, nxj) : st]
            ni, nj = xij.shape[-2:]
            result[:, :ni, :nj] += xij
            count[:ni, :nj] += 1

    result /= count

    # run simulation
    model = nengo.Network()
    with model:
        u = nengo.Node(image.ravel())
        v = nengo.Node(Pool2d(image.shape, s, strides=st))
        nengo.Connection(u, v, synapse=None)
        vp = nengo.Probe(v)

    with Simulator(model) as sim:
        sim.step()
    y = sim.data[vp][-1].reshape(result.shape)
    assert np.allclose(result, y, rtol=1e-3, atol=1e-6)
