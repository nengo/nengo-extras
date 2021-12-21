import gzip
import io
import os
import re
import tarfile

import nengo
import numpy as np
import PIL.Image

from nengo_extras.compat import is_integer, is_iterable, pickle_load_bytes, urlretrieve

data_dir = nengo.rc.get("nengo_extras", "data_dir")


def get_file(filename, url):
    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        print("Retrieving %r" % url)
        urlretrieve(url, filename=filename)
        print("Data retrieved as %r" % filename)
    return filename


def get_cifar10_tar_gz():
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    return get_file(filename, url)


def get_cifar100_tar_gz():
    filename = os.path.join(data_dir, "cifar-100-python.tar.gz")
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    return get_file(filename, url)


def get_ilsvrc2012_tar_gz():
    filename = os.path.join(data_dir, "ilsvrc-2012-batches-test3.tar.gz")
    url = "http://files.figshare.com/5370887/ilsvrc-2012-batches-test3.tar.gz"
    return get_file(filename, url)


def get_mnist_pkl_gz():
    filename = os.path.join(data_dir, "mnist.pkl.gz")
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    return get_file(filename, url)


def get_svhn_tar_gz():
    filename = os.path.join(data_dir, "svhn-py-colmajor.tar.gz")
    url = "https://files.figshare.com/7868377/svhn-py-colmajor.tar.gz"
    return get_file(filename, url)


def unpickle_tarfile(tar, name):
    tarextract = tar.extractfile(name)
    return pickle_load_bytes(tarextract)


def load_cifar10(filepath=None, n_train=5, n_test=1, label_names=False):
    """Load the CIFAR-10 dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'cifar-10-python.tar.gz' file.
        If ``None``, the file will be downloaded to the current directory.
    n_train : int (optional, Default: 6)
        The number of training batches to load (max: 6).
    n_test : int (optional, Default: 6)
        The number of testing batches to load (max: 1).
    label_names : boolean (optional, Default: False)
        Whether to provide the category label names.

    Returns
    -------
    train_set : (n_train, n_pixels) ndarray, (n_train,) ndarray
        A tuple of the training image array and label array.
    test_set : (n_test, n_pixels) ndarray, (n_test,) ndarray
        A tuple of the testing image array and label array.
    label_names : list
        A list of the label names.
    """
    if filepath is None:
        filepath = get_cifar10_tar_gz()

    # helper for reading each batch file
    def read_tar_batch(tar, name):
        data = unpickle_tarfile(tar, name)
        return data[b"data"], np.array(data[b"labels"])

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        if n_train < 1:
            train = (np.array([]), np.array([]))
        else:
            train = ([], [])
            for i in range(n_train):
                data, labels = read_tar_batch(
                    tar, "cifar-10-batches-py/data_batch_%d" % (i + 1)
                )
                train[0].append(data)
                train[1].append(labels)

            train = (np.vstack(train[0]), np.hstack(train[1]))

        if n_test < 1:
            test = (np.array([]), np.array([]))
        else:
            test = read_tar_batch(tar, "cifar-10-batches-py/test_batch")

        if label_names:
            meta = unpickle_tarfile(tar, "cifar-10-batches-py/batches.meta")
            names = meta[b"label_names"]

    return (train, test) + ((names,) if label_names else ())


def load_cifar100(filepath=None, fine_labels=True, label_names=False):
    """Load the CIFAR-100 dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'cifar-100-python.tar.gz' file.
        If ``None``, the file will be downloaded to the current directory.
    fine_labels : boolean (optional, Default: True)
        Whether to provide the fine labels or coarse labels.
    label_names : boolean (optional, Default: False)
        Whether to provide the category label names.

    Returns
    -------
    train_set : (n_train, n_pixels) ndarray, (n_train,) ndarray
        A tuple of the training image array and label array.
    test_set : (n_test, n_pixels) ndarray, (n_test,) ndarray
        A tuple of the testing image array and label array.
    label_names : list
        A list of the label names.
    """
    if filepath is None:
        filepath = get_cifar100_tar_gz()

    # helper for reading each batch file
    def read_tar_batch(tar, name):
        data = unpickle_tarfile(tar, name)
        return data[b"data"], np.array(
            data[b"fine_labels" if fine_labels else b"coarse_labels"]
        )

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        train = read_tar_batch(tar, "cifar-100-python/train")
        test = read_tar_batch(tar, "cifar-100-python/test")
        if label_names:
            meta = unpickle_tarfile(tar, "cifar-100-python/meta")
            names = meta[b"fine_label_names" if fine_labels else b"coarse_label_names"]

    return (train, test) + ((names,) if label_names else ())


def load_ilsvrc2012(filepath=None, n_files=None):
    """Load part of the ILSVRC 2012 (ImageNet) dataset.

    This loads a small section of the ImageNet Large Scale Visual Recognition
    Challenge (ILSVRC) 2012 dataset. The images are from the test portion of
    the dataset, and can be used to test pretrained classifiers.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'ilsvrc-2012-batches-test3.tar.gz'.
        If ``None``, the file will be downloaded to the current directory.
    n_files : int (optional, Default: None)
        Number of files (batches) to load from the archive. Defaults to all.

    Returns
    -------
    images : (n_images, nc, ny, nx) ndarray
        The loaded images. nc = number of channels, ny = height, nx = width
    labels : (n_images,) ndarray
        The labels of the images.
    data_mean : (nc, ny, nx) ndarray
        The mean of the images in the whole of the training set.
    label_names : list
        A list of the label names.
    """
    if filepath is None:
        filepath = get_ilsvrc2012_tar_gz()

    # helper for reading each batch file
    def read_tar_batch(tar, name):
        data = unpickle_tarfile(tar, name)
        return data[b"data"], data[b"labels"]  # JPEG strings, labels

    def bytes_to_array(b):
        image = PIL.Image.open(io.BytesIO(b))
        array = np.array(image, dtype=np.uint8).reshape(
            (image.size[0], image.size[1], 3)
        )
        array = np.transpose(array, (2, 0, 1))
        return array

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        names = tar.getnames()
        regex = re.compile(r".*/data_batch_([0-9]+\.[0-9]+)")
        matches = [regex.match(name) for name in names]
        matches = [match for match in matches if match]

        batchfiles = {}
        for match in matches:
            batchfiles[float(match.groups()[-1])] = match.group()

        raw_images = []
        raw_labels = []
        for key in sorted(list(batchfiles))[:n_files]:
            batchfile = batchfiles[key]
            x, y = read_tar_batch(tar, batchfile)
            raw_images.extend(x)
            raw_labels.extend(y)

        n_images = len(raw_images)
        image_shape = bytes_to_array(raw_images[0]).shape
        images = np.zeros((n_images,) + image_shape, dtype=np.uint8)
        for i, s in enumerate(raw_images):
            images[i] = bytes_to_array(s)

        labels = np.array(raw_labels)
        labels.shape = (n_images,)

        meta = unpickle_tarfile(tar, "batches.meta")
        data_mean = meta[b"data_mean"].reshape(image_shape)
        label_names = meta[b"label_names"]

    return images, labels, data_mean, label_names


def load_ilsvrc2012_metadata(filepath=None):
    if filepath is None:
        filepath = get_ilsvrc2012_tar_gz()

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        meta = unpickle_tarfile(tar, "batches.meta")
        data_mean = meta[b"data_mean"].reshape((3, 256, 256))
        label_names = meta[b"label_names"]

    return data_mean, label_names


def load_mnist(filepath=None, validation=False):
    """Load the MNIST dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'mnist.pkl.gz' file.
        If ``None``, the file will be downloaded to the current directory.
    validation : boolean (optional, Default: False)
        Whether to provide the validation data as a separate set (True),
        or combine it into the training data (False).

    Returns
    -------
    train_set : (n_train, n_pixels) ndarray, (n_train,) ndarray
        A tuple of the training image array and label array.
    validation_set : (n_valid, n_pixels) ndarray, (n_valid,) ndarray
        A tuple of the validation image array and label array
        (if ``validation``)
    test_set : (n_test, n_pixels) ndarray, (n_test,) ndarray
        A tuple of the testing image array and label array.
    """
    if filepath is None:
        filepath = get_mnist_pkl_gz()

    filepath = os.path.expanduser(filepath)
    with gzip.open(filepath, "rb") as f:
        train_set, valid_set, test_set = pickle_load_bytes(f)

    if validation:
        return train_set, valid_set, test_set
    # combine valid into train
    train_set = (
        np.vstack((train_set[0], valid_set[0])),
        np.hstack((train_set[1], valid_set[1])),
    )
    return train_set, test_set


def load_svhn(filepath=None, n_train=9, n_test=3, data_mean=False, label_names=False):
    """Load the SVHN dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'svhn-py-colmajor.tar.gz' file.
        If ``None``, the file will be downloaded to the current directory.
    n_train : int (optional, Default: 6)
        The number of training batches to load (max: 6).
    n_test : int (optional, Default: 6)
        The number of testing batches to load (max: 1).
    label_names : boolean (optional, Default: False)
        Whether to provide the category label names.

    Returns
    -------
    train_set : (n_train, n_pixels) ndarray, (n_train,) ndarray
        A tuple of the training image array and label array.
    test_set : (n_test, n_pixels) ndarray, (n_test,) ndarray
        A tuple of the testing image array and label array.
    label_names : list
        A list of the label names.
    """
    shape = (3, 32, 32)

    if filepath is None:
        filepath = get_svhn_tar_gz()

    def read_tar_batch(tar, name):
        data = unpickle_tarfile(tar, name)
        return data[b"data"], np.array(data[b"labels"])

    def load_batches(tar, inds):
        if len(inds) < 1:
            return (np.array([]), np.array([]))

        batches = ([], [])
        for i in inds:
            data, labels = read_tar_batch(tar, "svhn-py-colmajor/data_batch_%d" % i)
            batches[0].append(data.T)
            batches[1].append(labels)

        return (np.vstack(batches[0]).reshape((-1,) + shape), np.hstack(batches[1]))

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        train = load_batches(tar, list(range(1, n_train + 1)))
        test = load_batches(tar, list(range(10, n_test + 10)))

        if label_names or data_mean:
            meta = unpickle_tarfile(tar, "svhn-py-colmajor/batches.meta")
        data_mean = (meta[b"data_mean"].reshape(shape),) if data_mean else ()
        label_names = (meta[b"label_names"],) if label_names else ()

    return (train, test) + data_mean + label_names


def spasafe_name(name, pre_comma_only=True):
    """Make a name safe to use as a SPA semantic pointer name.

    Ensure a name conforms with SPA name standards. Replaces hyphens and
    spaces with underscores, removes all other characters, and makes the
    first letter uppercase.

    Parameters
    ----------
    pre_comma_only : boolean
        Only use the part of a name before a/the first comma.
    """
    if len(name) == 0:
        raise ValueError("Empty name.")

    if pre_comma_only and "," in name:
        name = name.split(",")[0]  # part before first comma
    name = name.strip()
    name = re.sub(r"(\s|-|,)+", "_", name)  # repl space/hyphen/comma w undersc
    name = re.sub("(^[^a-zA-Z]+)|[^a-zA-Z0-9_]+", "", name)  # del other chars
    name = name[0].upper() + name[1:]  # capitalize first letter
    return name


def spasafe_names(label_names, pre_comma_only=True):
    """Make names safe to use as SPA semantic pointer names.

    Format a list of names to conform with SPA name standards. In addition
    to running each name through ``spasafe_name``, this function numbers
    duplicate names so they are unique.

    Parameters
    ----------
    pre_comma_only : boolean
        Only use the part of a name before a/the first comma.
    """
    vocab_names = [
        spasafe_name(name, pre_comma_only=pre_comma_only) for name in label_names
    ]

    # number duplicates
    unique = set()
    duplicates = []
    for name in vocab_names:
        if name in unique:
            duplicates.append(name)
        else:
            unique.add(name)

    duplicates = {name: 0 for name in duplicates}
    for i, name in enumerate(vocab_names):
        if name in duplicates:
            vocab_names[i] = "%s%d" % (name, duplicates[name])
            duplicates[name] += 1

    return vocab_names


def one_hot_from_labels(labels, classes=None, dtype=float):
    """Turn integer labels into a one-hot encoding.

    Parameters
    ==========
    labels : (n,) array
        Labels to turn into one-hot encoding.
    classes : int or (n_classes,) array (optional)
        Classes for encoding. If integer and ``labels.dtype`` is integer, this
        is the number of classes in the encoding. If iterable, this is the
        list of classes to place in the one-hot (must be a superset of the
        unique elements in ``labels``).
    dtype : dtype (optional)
        Data type of returned one-hot encoding (defaults to ``float``).
    """
    assert labels.ndim == 1
    n = labels.shape[0]

    if np.issubdtype(labels.dtype, np.integer) and (
        classes is None or is_integer(classes)
    ):
        index = labels
        index_min, index_max = index.min(), index.max()
        n_classes = (index_max + 1) if classes is None else classes
        assert index_min >= 0
        assert index_max < n_classes
    else:
        if classes is not None:
            assert is_iterable(classes)
            assert set(np.unique(labels)).issubset(classes)
        classes = np.unique(labels) if classes is None else classes
        n_classes = len(classes)

        c_index = np.argsort(classes)
        c_sorted = classes[c_index]
        index = c_index[np.searchsorted(c_sorted, labels)]

    y = np.zeros((n, n_classes), dtype=dtype)
    y[np.arange(n), index] = 1
    return y


class ZCAWhiten:
    """ZCA Whitening

    References
    ----------
    .. [1] Krizhevsky, Alex. "Learning multiple layers of features from tiny
           images" (2009) MSc Thesis, Dept. of Comp. Science, Univ. of
           Toronto. pp. 48-49.
    """

    def __init__(self, beta=1e-2, gamma=1e-5):
        self.beta = beta
        self.gamma = gamma

        self.dims = None
        self.pixel_mu = None
        self.e = None
        self.V = None
        self.Sinv = None

    def contrast_normalize(self, X, remove_mean=True, beta=None, hard_beta=True):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("contrast_normalize requires flat patches")

        Xc = X - X.mean(axis=1)[:, None] if remove_mean else X
        l2 = (Xc * Xc).sum(axis=1)

        beta = self.beta if beta is None else beta
        div2 = np.maximum(l2, beta) if hard_beta else l2 + beta
        return Xc / np.sqrt(div2[:, None])

    def fit(self, X):
        """Fit whitening transform to training data

        Parameters
        ----------
        X : array_like
            Flattened data, with each row corresponding to one example
        """
        X = self.contrast_normalize(X)
        self.dims = X.shape[1]

        self.pixel_mu = X.mean(axis=0)
        X -= self.pixel_mu[None, :]  # each pixel has zero mean

        S = np.dot(X.T, X) / (X.shape[0] - 1)
        e, V = np.linalg.eigh(S)
        self.e = e
        self.V = V

        self.Sinv = np.dot(np.sqrt(1.0 / (e + self.gamma)) * V, V.T)

        return np.dot(X, self.Sinv)

    def transform(self, X):
        assert self.dims is not None

        X = self.contrast_normalize(X, beta=self.beta)
        assert X.shape[1] == self.dims

        X -= self.pixel_mu[None, :]
        return np.dot(X, self.Sinv)
