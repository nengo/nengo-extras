import numpy as np

from nengo_extras.data import one_hot_from_labels


def test_one_hot_from_labels_int(rng):
    nc = 19
    labels = rng.randint(nc, size=1000)

    yref = np.zeros((len(labels), nc))
    yref[np.arange(len(labels)), labels] = 1

    y0 = one_hot_from_labels(labels)
    y1 = one_hot_from_labels(labels, classes=nc+5)
    assert np.array_equal(y0, yref)
    assert np.array_equal(y0, y1[:, :nc])
    assert (y1[:, nc:] == 0).all()


def test_one_hot_from_labels_skip(rng):
    labels = 2*rng.randint(4, size=1000)

    yref = np.zeros((len(labels), labels.max()+1))
    yref[np.arange(len(labels)), labels] = 1
    y = one_hot_from_labels(labels)
    assert np.array_equal(y, yref)


def test_one_hot_from_labels_float(rng):
    classes = rng.uniform(0, 9, size=11)
    inds = rng.randint(len(classes), size=1000)
    labels = classes[inds]

    yref = np.zeros((len(labels), len(classes)))
    yref[np.arange(len(labels)), inds] = 1

    y = one_hot_from_labels(labels, classes=classes)
    assert np.array_equal(y, yref)
