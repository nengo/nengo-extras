import gzip
import os
import urllib

import numpy as np


def load_mnist(data_dir="."):
    fnames = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    url = 'http://yann.lecun.com/exdb/mnist/'
    for fname in fnames:
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            print("MNIST file '%s' not found in directory '%s'. Downloading."
                  % (fname, data_dir))
            urllib.urlretrieve(url+fname, fpath)

    data = []
    for fname in fnames:
        fd = gzip.open(os.path.join(data_dir, fname))
        data.append(np.fromstring(fd.read(), dtype=np.uint8))

    trX = data[0][16:].reshape((60000, -1)) / 255.
    trX = trX.reshape(-1, 28, 28)

    trYlabels = data[1][8:].reshape(60000)
    trY = np.zeros((60000, 10))
    trY[np.arange(60000), trYlabels] = 1

    teX = data[2][16:].reshape((10000, -1)) / 255.
    teX = teX.reshape(-1, 28, 28)

    teYlabels = data[3][8:].reshape(10000)
    teY = np.zeros((10000, 10))
    teY[np.arange(10000), teYlabels] = 1

    return trX, teX, trY, teY

trX, teX, trY, teY = load_mnist()
