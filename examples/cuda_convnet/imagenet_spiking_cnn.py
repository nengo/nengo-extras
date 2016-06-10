"""
Classifier for the ImageNet ILSVRC-2012 dataset.
"""
import os

import nengo
import numpy as np

from nengo_extras.data import load_ilsvrc2012, spasafe_names
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle

# retrieve from https://figshare.com/s/cdde71007405eb11a88f
filename = 'ilsvrc-2012-batches-test3.tar.gz'
X_test, Y_test, data_mean, label_names = load_ilsvrc2012(filename, n_files=1)

X_test = X_test.astype('float32')

# crop data
X_test = X_test[:, :, 16:-16, 16:-16]

# subtract mean
data_mean = X_test.mean(axis=0)
X_test -= data_mean

# retrieve from https://figshare.com/s/f343c68df647e675af28
cc_model = load_model_pickle('ilsvrc2012-lif-48.pkl')


# --- Run model in Nengo
presentation_time = 0.2

model = nengo.Network()
with model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    ccnet = CudaConvnetNetwork(cc_model, synapse=nengo.synapses.Alpha(0.001))
    nengo.Connection(u, ccnet.inputs['data'], synapse=None)

    input_p = nengo.Probe(u)
    output_p = nengo.Probe(ccnet.output)

    # --- image display
    # input_shape = kmodel.input_shape[1:]
    # input_shape = (3, 24, 24)
    input_shape = (3, 224, 224)

    def display_func(t, x, input_shape=input_shape):
        import base64
        import PIL
        import cStringIO

        values = x.reshape(input_shape)
        values = values + data_mean
        values = values.transpose((1, 2, 0))
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

with nengo.Simulator(model) as sim:
    nb_presentations = 20
    sim.run(nb_presentations * presentation_time)

nt = int(presentation_time / sim.dt)
blocks = sim.data[output_p].reshape(nb_presentations, nt, nb_classes)
choices = np.argmax(blocks[:, -20:, :].mean(axis=1), axis=1)
accuracy = (choices == y_test[:nb_presentations]).mean()
print('Spiking accuracy (%d examples): %0.3f' % (nb_presentations, accuracy))
