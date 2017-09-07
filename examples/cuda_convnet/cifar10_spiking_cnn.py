from __future__ import print_function

import os

import nengo
import nengo_ocl
import numpy as np

from nengo_extras.data import load_cifar10
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle
from nengo_extras.gui import image_display_function


(X_train, y_train), (X_test, y_test), label_names = load_cifar10(label_names=True)
X_train = X_train.reshape(-1, 3, 32, 32).astype('float32')
X_test = X_test.reshape(-1, 3, 32, 32).astype('float32')
n_classes = len(label_names)

# crop data
X_train = X_train[:, :, 4:-4, 4:-4]
X_test = X_test[:, :, 4:-4, 4:-4]

# subtract mean
data_mean = X_train.mean(axis=0)
X_train -= data_mean
X_test -= data_mean

# retrieve from https://figshare.com/s/49741f9e2d0d29f68871
cc_model = load_model_pickle('cifar10-lif-1628.pkl')


# --- Run model in Nengo
presentation_time = 0.2

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.005))
    nengo.Connection(u, ccnet.input, synapse=None)

    input_p = nengo.Probe(u)
    output_p = nengo.Probe(ccnet.output)

    # --- image display
    image_shape = X_test.shape[1:]
    display_f = image_display_function(image_shape, scale=1, offset=data_mean)
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
    nengo.Connection(ccnet.output, output.input)


n_presentations = 100

if 0:
    # run ANN in Theano
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
    Q = ccnet.theano_compute(X_test[:n_presentations])
    Z = np.argmax(Q, axis=-1) == y_test[:n_presentations]
    print("ANN accuracy (%d examples): %0.4f" % (n_presentations, Z.mean()))


with nengo_ocl.Simulator(model) as sim:
    sim.run(n_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
blocks = sim.data[output_p].reshape(n_presentations, nt, n_classes)
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == y_test[:n_presentations]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (n_presentations, accuracy))
