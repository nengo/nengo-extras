"""
Generate Gabor filters for a set of images based on the statistics
of those images.
"""

import matplotlib.pyplot as plt
import numpy as np

from nengo.dists import Uniform
from nengo_extras.data import load_mnist, patches_from_images
from nengo_extras.vision import Gabor, gabors_for_images, gabors_for_patches


images = load_mnist('~/data/mnist.pkl.gz')[0][0].reshape(-1, 28, 28)
# images = Gabor(theta=Uniform(-0.1, 0.1), freq=Uniform(0.5, 1.5)).generate(10000, (28, 28))
# images = Gabor(theta=Uniform(-0.1, 0.1), freq=Uniform(2., 3.)).generate(10000, (28, 28))
# images = Gabor(theta=Uniform(-0.1, 0.1), freq=Uniform(1., 2.)).generate(10000, (28, 28))
# images = Gabor(theta=Uniform(-0.1, 0.1), freq=Uniform(5., 6.)).generate(10000, (28, 28))

patches = patches_from_images(images, 10000, (11, 11))

gabors1 = gabors_for_images(images, 1000, images.shape[-2:])
gabors2 = gabors_for_images(images, 1000, (11, 11))
# gabors2 = gabors_for_patches(images, 1000, (11, 11))

def spectrum(images):
    F = np.fft.fft2(images)
    Fmean = np.abs(F).mean(0)
    Fmean[0, 0] = 0
    return np.fft.fftshift(Fmean)

plt.figure()
plt.subplot(221)
plt.imshow(spectrum(images), interpolation='none')
plt.subplot(222)
plt.imshow(spectrum(patches), interpolation='none')
plt.subplot(223)
plt.imshow(spectrum(gabors1), interpolation='none')
plt.subplot(224)
plt.imshow(spectrum(gabors2), interpolation='none')
plt.show()
