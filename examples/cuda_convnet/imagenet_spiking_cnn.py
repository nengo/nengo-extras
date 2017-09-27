"""
Classifier for the ImageNet ILSVRC-2012 dataset.
"""
import os

import nengo
import nengo_ocl
import numpy as np

from nengo_extras.data import load_ilsvrc2012, spasafe_names
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle
from nengo_extras.gui import image_display_function

X_test, Y_test, data_mean, label_names = load_ilsvrc2012(n_files=1)
X_test = X_test.astype('float32')

# crop data
X_test = X_test[:, :, 16:-16, 16:-16]
data_mean = data_mean[:, 16:-16, 16:-16]
image_shape = X_test.shape[1:]

# subtract mean
X_test -= data_mean

# retrieve from https://figshare.com/s/f343c68df647e675af28
cc_model = load_model_pickle('ilsvrc2012-lif-48.pkl')


# --- Run model in Nengo
presentation_time = 0.2

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.001))
    nengo.Connection(u, ccnet.input, synapse=None)

    # input_p = nengo.Probe(u)
    output_p = nengo.Probe(ccnet.output)

    # --- image display
    display_f = image_display_function(image_shape, scale=1., offset=data_mean)
    display_node = nengo.Node(display_f, size_in=u.size_out)
    nengo.Connection(u, display_node, synapse=None)

    # --- output spa display
    vocab_names = spasafe_names(label_names)
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
    choices = np.argsort(Q, axis=1)
    top5corrects = choices[:, -5:] == Y_test[:n_presentations, None]
    top1accuracy = top5corrects[:, -1].mean()
    top5accuracy = np.any(top5corrects, axis=1).mean()
    print("ANN accuracy (%d examples): %0.3f (top-1), %0.3f (top-5)" %
          (n_presentations, top1accuracy, top5accuracy))


with nengo_ocl.Simulator(model) as sim:
    sim.run(n_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
n_classes = ccnet.output.size_out
blocks = sim.data[output_p].reshape(n_presentations, nt, n_classes)
choices = np.argsort(blocks[:, -20:, :].mean(axis=1), axis=1)
top5corrects = choices[:, -5:] == Y_test[:n_presentations, None]
top1accuracy = top5corrects[:, -1].mean()
top5accuracy = np.any(top5corrects, axis=1).mean()
print('Spiking accuracy (%d examples): %0.3f (top-1), %0.3f (top-5)' %
      (n_presentations, top1accuracy, top5accuracy))
