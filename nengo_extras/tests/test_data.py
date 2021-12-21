import numpy as np
import pytest

from nengo_extras.data import (
    load_cifar10,
    load_cifar100,
    load_ilsvrc2012,
    load_mnist,
    load_svhn,
    one_hot_from_labels,
    spasafe_name,
    spasafe_names,
)
from nengo_extras.matplotlib import tile


@pytest.mark.slow
def test_load_cifar10(plt):
    trainX = load_cifar10(n_train=1, n_test=0)[0][0]
    trainX = trainX.reshape((-1, 3, 32, 32))
    trainX = np.transpose(trainX, (0, 2, 3, 1))
    tile(trainX, ax=plt.gca())


@pytest.mark.slow
def test_load_cifar100(plt):
    trainX = load_cifar100()[0][0]
    trainX = trainX.reshape((-1, 3, 32, 32))
    trainX = np.transpose(trainX, (0, 2, 3, 1))
    tile(trainX, ax=plt.gca())


@pytest.mark.slow
def test_load_ilsvrc2012(plt):
    testX, _, _, _ = load_ilsvrc2012(n_files=1)
    testX = testX.reshape((-1, 3, 256, 256))
    testX = np.transpose(testX, (0, 2, 3, 1))
    tile(testX, ax=plt.gca())


@pytest.mark.slow
def test_load_mnist(plt):
    trainX = load_mnist()[0][0]
    tile(trainX.reshape((-1, 28, 28)), ax=plt.gca())


@pytest.mark.slow
def test_load_svhn(plt):
    (trainX, _), (_, _) = load_svhn()
    trainX = trainX.reshape((-1, 3, 32, 32))
    trainX = np.transpose(trainX, (0, 2, 3, 1))
    tile(trainX, ax=plt.gca())


def test_one_hot_from_labels_int(rng):
    nc = 19
    labels = rng.randint(nc, size=1000)

    yref = np.zeros((len(labels), nc))
    yref[np.arange(len(labels)), labels] = 1

    y0 = one_hot_from_labels(labels)
    y1 = one_hot_from_labels(labels, classes=nc + 5)
    assert np.array_equal(y0, yref)
    assert np.array_equal(y0, y1[:, :nc])
    assert (y1[:, nc:] == 0).all()


def test_one_hot_from_labels_skip(rng):
    labels = 2 * rng.randint(4, size=1000)

    yref = np.zeros((len(labels), labels.max() + 1))
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


def test_spasafe_name():
    assert spasafe_name("UPPER") == "UPPER"
    assert spasafe_name("Camel") == "Camel"
    assert spasafe_name("lower") == "Lower"
    assert spasafe_name("Under_score") == "Under_score"
    assert (
        spasafe_name("Weird.,:[]^!<>=&'\"symbols", pre_comma_only=False)
        == "Weird_symbols"
    )
    assert spasafe_name("Weird.,:[]^!<>=&'\"symbols", pre_comma_only=True) == "Weird"
    assert spasafe_name("  \tWhite- \t\r\nspace  \r\n") == "White_space"
    assert spasafe_name("multiple correction's") == "Multiple_corrections"
    assert spasafe_name("123four") == "Four"
    assert spasafe_name("A") == "A"
    with pytest.raises(ValueError):
        spasafe_name("")


def test_spasafe_names():
    assert spasafe_names(["A,B", "A", "c's"]) == ["A0", "A1", "Cs"]
