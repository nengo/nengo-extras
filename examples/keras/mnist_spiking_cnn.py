from __future__ import print_function

import os

import nengo
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D
from keras.utils import np_utils

import nengo
from nengo_extras.keras import (
    load_model_pair, save_model_pair, SequentialNetwork, SoftLIF)


filename = 'mnist_spiking_cnn'

# --- Load data
img_rows, img_cols = 28, 28
nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# --- Train model
if not os.path.exists(filename + '.h5'):
    batch_size = 128
    nb_epoch = 12

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    kmodel = Sequential()

    softlif_params = dict(
        sigma=0.02, amplitude=0.063, tau_rc=0.022, tau_ref=0.002)

    kmodel.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                             border_mode='valid',
                             input_shape=(1, img_rows, img_cols)))
    kmodel.add(SoftLIF(**softlif_params))
    kmodel.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    kmodel.add(SoftLIF(**softlif_params))
    kmodel.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
    kmodel.add(Dropout(0.25))

    kmodel.add(Flatten())
    kmodel.add(Dense(128))
    kmodel.add(SoftLIF(**softlif_params))
    kmodel.add(Dropout(0.5))
    kmodel.add(Dense(nb_classes))
    kmodel.add(Activation('softmax'))

    kmodel.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['accuracy'])

    kmodel.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1, validation_data=(X_test, Y_test))
    score = kmodel.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    save_model_pair(kmodel, filename, overwrite=True)

else:
    kmodel = load_model_pair(filename)


# --- Run model in Nengo
presentation_time = 0.2

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    seq = SequentialNetwork(kmodel, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, seq.input, synapse=None)

    input_p = nengo.Probe(u)
    output_p = nengo.Probe(seq.output)

    # --- image display
    input_shape = kmodel.input_shape[1:]

    def display_func(t, x, input_shape=input_shape):
        import base64
        import PIL.Image
        import cStringIO

        values = x.reshape(input_shape)
        values = values.transpose((1, 2, 0))
        values = values * 255.
        values = values.astype('uint8')

        if values.shape[-1] == 1:
            values = values[:, :, 0]

        png = PIL.Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())

        display_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    display_node = nengo.Node(display_func, size_in=u.size_out)
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
    nengo.Connection(seq.output, output.input)

with nengo.Simulator(model) as sim:
    nb_presentations = 20
    sim.run(nb_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
blocks = sim.data[output_p].reshape(nb_presentations, nt, nb_classes)
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == y_test[:nb_presentations]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (nb_presentations, accuracy))
