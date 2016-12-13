from __future__ import print_function

import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

import numpy as np
np.random.seed(125)

import keras
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Flatten, Convolution2D, AveragePooling2D,
    LocallyConnected2D, ZeroPadding2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import nengo
import nengo_ocl
from nengo_extras.data import load_cifar10
from nengo_extras.keras import (
    load_model_pair, save_model_pair, SequentialNetwork, SoftLIF, LSUVinit)
from nengo_extras.gui import image_display_function
from nengo_extras.neurons import softplus

filename = 'cifar10_spiking_cnn'

# --- Load data
img_shape = (3, 32, 32)
n_classes = 10

(X_train, y_train), (X_test, y_test), label_names = load_cifar10(
    label_names=True)

preprocess = (
    lambda X: X.reshape((X.shape[0],) + img_shape).astype('float32') / 255)
X_train, X_test = preprocess(X_train), preprocess(X_test)

data_mean = np.zeros_like(X_train[0])
# data_mean = X_train.mean(axis=0)
# X_train -= data_mean
# X_test -= data_mean

# --- Train model
if not os.path.exists(filename + '.h5'):
    # batch_size = 128
    batch_size = 32

    # n_epochs = 12
    # n_epochs = 100
    n_epochs = 200

    softlif_params = dict(
        sigma=0.02, amplitude=0.063, tau_rc=0.05, tau_ref=0.001,
        noise=('gaussian', 10))

    # act_layer = lambda: Activation('relu')
    act_layer = lambda: SoftLIF(**softlif_params)

    kmodel = Sequential()
    kmodel.add(Convolution2D(64, 3, 3, input_shape=img_shape))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))
    kmodel.add(Convolution2D(64, 3, 3, subsample=(2, 2)))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))
    kmodel.add(Convolution2D(128, 3, 3))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))
    kmodel.add(Convolution2D(128, 3, 3, subsample=(2, 2)))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))
    kmodel.add(Convolution2D(256, 3, 3))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))

    kmodel.add(Flatten())
    kmodel.add(Dense(2048))
    kmodel.add(act_layer())
    kmodel.add(Dropout(0.25))

    kmodel.add(Dense(n_classes))
    kmodel.add(Activation('softmax'))

    lr = 1e-2
    optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9, decay=lr/n_epochs)
    kmodel.compile(loss='categorical_crossentropy',
                   optimizer=optimizer,
                   metrics=['accuracy'])

    for layer in kmodel.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            print(weights[0].shape)

    LSUVinit(kmodel, X_train[:100])

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    gen_train = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    kmodel.fit_generator(
        gen_train.flow(X_train, Y_train, batch_size=batch_size),
        samples_per_epoch=len(X_train), nb_epoch=n_epochs,
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
    net = SequentialNetwork(kmodel, synapse=nengo.synapses.Alpha(0.003))
    nengo.Connection(u, net.input, synapse=None)

    input_p = nengo.Probe(u)
    output_p = nengo.Probe(net.output)

    # --- image display
    image_shape = X_test.shape[1:]
    display_f = image_display_function(image_shape, scale=255, offset=data_mean)
    display_node = nengo.Node(display_f, size_in=u.size_out)
    nengo.Connection(u, display_node, synapse=None)

    # --- output spa display
    vocab_names = [s.upper() for s in label_names]
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output = nengo.spa.State(len(vocab_names), subdimensions=10, vocab=vocab)
    nengo.Connection(net.output, output.input)


n_presentations = 100

if 0:
    # run ANN in Theano
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
    Q = ccnet.theano_compute(X_test[:n_presentations])
    Z = np.argmax(Q, axis=-1) == y_test[:n_presentations]
    print("ANN accuracy (%d examples): %0.3f" % (n_presentations, Z.mean()))


with nengo_ocl.Simulator(model) as sim:
    sim.run(n_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
blocks = sim.data[output_p].reshape(n_presentations, nt, n_classes)
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == y_test[:n_presentations]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_presentations, accuracy))
