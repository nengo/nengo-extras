from __future__ import print_function

import os

import nengo
import nengo_ocl
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Flatten, Convolution2D, AveragePooling2D)
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils

from nengo_extras.keras import (
    load_model_pair, save_model_pair, SequentialNetwork, SoftLIF)
from nengo_extras.gui import image_display_function

np.random.seed(1)
filename = 'mnist_spiking_cnn'

# --- Load data
img_rows, img_cols = 28, 28
n_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
data_format = 'channels_first'


def preprocess(X):
    X = X.astype('float32')/128 - 1
    if data_format == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)

    return X

X_train, X_test = preprocess(X_train), preprocess(X_test)

# --- Train model
if not os.path.exists(filename + '.h5'):
    batch_size = 128
    epochs = 6

    n_filters = 32        # number of convolutional filters to use
    kernel_size = (3, 3)  # shape of each convolutional filter

    softlif_params = dict(
        sigma=0.01, amplitude=0.063, tau_rc=0.022, tau_ref=0.002)

    input_shape = X_train.shape[1:]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    # construct Keras model
    kmodel = Sequential()
    kmodel.add(GaussianNoise(0.1, input_shape=input_shape))

    kmodel.add(Convolution2D(n_filters, kernel_size, padding='valid',
                             strides=(2, 2),
                             data_format=data_format))
    kmodel.add(SoftLIF(**softlif_params))

    kmodel.add(Convolution2D(n_filters, kernel_size,
                             strides=(2, 2),
                             data_format=data_format))
    kmodel.add(SoftLIF(**softlif_params))

    kmodel.add(Flatten())
    kmodel.add(Dense(512))
    kmodel.add(SoftLIF(**softlif_params))
    kmodel.add(Dropout(0.5))

    kmodel.add(Dense(n_classes))
    kmodel.add(Activation('softmax'))

    # compile and fit Keras model
    optimizer = keras.optimizers.Nadam()
    kmodel.compile(loss='categorical_crossentropy',
                   optimizer=optimizer,
                   metrics=['accuracy'])
    kmodel.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
               verbose=1, validation_data=(X_test, Y_test))

    score = kmodel.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    save_model_pair(kmodel, filename, overwrite=True)

else:
    kmodel = load_model_pair(filename)


# --- Run model in Nengo
presentation_time = 0.15

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    knet = SequentialNetwork(kmodel, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, knet.input, synapse=None)

    input_p = nengo.Probe(u)
    output_p = nengo.Probe(knet.output)

    # --- image display
    image_shape = kmodel.input_shape[1:]
    display_f = image_display_function(image_shape)
    display_node = nengo.Node(display_f, size_in=u.size_out)
    nengo.Connection(u, display_node, synapse=None)

    # --- output spa display
    vocab_names = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
                   'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output = nengo.spa.State(len(vocab_names), subdimensions=10, vocab=vocab)
    nengo.Connection(knet.output, output.input)


n_presentations = 100

if 0:
    # run ANN in Theano
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
    Q = knet.theano_compute(X_test[:n_presentations])
    Z = np.argmax(Q, axis=-1) == y_test[:n_presentations]
    print("ANN accuracy (%d examples): %0.3f" % (n_presentations, Z.mean()))


with nengo_ocl.Simulator(model) as sim:
    sim.run(n_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
blocks = sim.data[output_p].reshape(n_presentations, nt, n_classes)
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == y_test[:n_presentations]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_presentations, accuracy))
