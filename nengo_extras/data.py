import gzip
import os
import re
import tarfile
import urllib

from nengo.utils.compat import pickle
import numpy as np


def get_file(filename, url):
    if not os.path.exists(filename):
        print("Retrieving %r" % url)
        urllib.urlretrieve(url, filename=filename)
        print("Data retrieved as %r" % filename)
    return filename


def get_cifar10_tar_gz():
    filename = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    return get_file(filename, url)


def get_cifar100_tar_gz():
    filename = 'cifar-100-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    return get_file(filename, url)


def get_ilsvrc_tar_gz():
    filename = 'ilsvrc-2012-batches-test3.tar.gz'
    url = 'http://files.figshare.com/5370887/ilsvrc-2012-batches-test3.tar.gz'
    return get_file(filename, url)


def get_mnist_pkl_gz():
    filename = 'mnist.pkl.gz'
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    return get_file(filename, url)


def unpickle_tarfile(tar, name):
    tarextract = tar.extractfile(name)
    return pickle.load(tarextract)


def load_cifar10(filepath=None, n_train=5, n_test=1, label_names=False):
    """Load the CIFAR-10 dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'cifar-10-python.tar.gz' file.
        If `None`, the file will be downloaded to the current directory.
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
        return data['data'], np.array(data['labels'])

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:

        if n_train < 1:
            train = (np.array([]), np.array([]))
        else:
            train = ([], [])
            for i in range(n_train):
                data, labels = read_tar_batch(
                    tar, 'cifar-10-batches-py/data_batch_%d' % (i+1))
                train[0].append(data)
                train[1].append(labels)

            train = (np.vstack(train[0]), np.hstack(train[1]))

        if n_test < 1:
            test = (np.array([]), np.array([]))
        else:
            test = read_tar_batch(tar, 'cifar-10-batches-py/test_batch')

        if label_names:
            meta = unpickle_tarfile(tar, 'cifar-10-batches-py/batches.meta')
            names = meta['label_names']

    return (train, test) + ((names,) if label_names else ())


def load_cifar100(filepath=None, fine_labels=True, label_names=False):
    """Load the CIFAR-100 dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'cifar-100-python.tar.gz' file.
        If `None`, the file will be downloaded to the current directory.
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
        return data['data'], np.array(
            data['fine_labels' if fine_labels else 'coarse_labels'])

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:
        train = read_tar_batch(tar, 'cifar-100-python/train')
        test = read_tar_batch(tar, 'cifar-100-python/test')
        if label_names:
            meta = unpickle_tarfile(tar, 'cifar-100-python/meta')
            names = meta[
                'fine_label_names' if fine_labels else 'coarse_label_names']

    return (train, test) + ((names,) if label_names else ())


def load_ilsvrc2012(filepath=None, n_files=None):
    """Load the CIFAR-10 dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'cifar-10-python.tar.gz' file.
        If `None`, the file will be downloaded to the current directory.
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
    import PIL.Image
    try:
        from cStringIO import StringIO
    except:
        from io import StringIO

    if filepath is None:
        filepath = get_ilsvrc_tar_gz()

    # helper for reading each batch file
    def read_tar_batch(tar, name):
        data = unpickle_tarfile(tar, name)
        return data['data'], data['labels']  # JPEG strings, labels

    def string_to_array(s):
        f = StringIO(s)
        image = PIL.Image.open(f)
        array = np.array(image, dtype=np.uint8).reshape(
            image.size[0], image.size[1], 3)
        array = np.transpose(array, (2, 0, 1))
        return array

    filepath = os.path.expanduser(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:
        names = tar.getnames()
        regex = re.compile('.*/data_batch_([0-9]+\.[0-9]+)')
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

        image0 = string_to_array(raw_images[0])
        images = np.zeros((len(raw_images),) + image0.shape, dtype=np.uint8)
        for i, s in enumerate(raw_images):
            images[i] = string_to_array(s)

        labels = np.array(raw_labels)

        meta = unpickle_tarfile(tar, 'batches.meta')
        data_mean = meta['data_mean']
        label_names = meta['label_names']

    return images, labels, data_mean, label_names


def load_mnist(filepath=None, validation=False):
    """Load the MNIST dataset.

    Parameters
    ----------
    filepath : str (optional, Default: None)
        Path to the previously downloaded 'mnist.pkl.gz' file.
        If `None`, the file will be downloaded to the current directory.
    validation : boolean (optional, Default: False)
        Whether to provide the validation data as a separate set (True),
        or combine it into the training data (False).

    Returns
    -------
    train_set : (n_train, n_pixels) ndarray, (n_train,) ndarray
        A tuple of the training image array and label array.
    validation_set : (n_valid, n_pixels) ndarray, (n_valid,) ndarray
        A tuple of the validation image array and label array (if `validation`)
    test_set : (n_test, n_pixels) ndarray, (n_test,) ndarray
        A tuple of the testing image array and label array.
    """
    if filepath is None:
        filepath = get_mnist_pkl_gz()

    filepath = os.path.expanduser(filepath)
    with gzip.open(filepath, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    if validation:
        return train_set, valid_set, test_set
    else:  # combine valid into train
        train_set = (np.vstack((train_set[0], valid_set[0])),
                     np.hstack((train_set[1], valid_set[1])))
        return train_set, test_set


def spasafe_names(label_names):
    vocab_names = [
        (name.split(',')[0] if ',' in name else name).upper().replace(' ', '_')
        for i, name in enumerate(label_names)]

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
            vocab_names[i] = '%s%d' % (name, duplicates[name])
            duplicates[name] += 1

    return vocab_names
