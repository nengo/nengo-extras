"""
Single-layer NEF network applied to MNIST
"""
import nengo
import numpy as np

import nengo_extras.solvers
from nengo_extras.data import load_mnist, one_hot_from_labels
from nengo_extras.vision import Gabor, Mask
from nengo_extras.gui import image_display_function

# --- options
rng = np.random.RandomState(0)

# solver = nengo.solvers.LstsqL2(reg=0.01)
solver = nengo_extras.solvers.LstsqClassifier(reg=0.02)
# solver = nengo_extras.solvers.SoftmaxClassifier(reg=0.004)

n_hid = 1000
# n_hid = 2000
# n_hid = 5000

# --- load the data
img_shape = (28, 28)

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2.0*X_train - 1  # normalize to -1 to 1
X_test = 2.0*X_test - 1  # normalize to -1 to 1

T_train = one_hot_from_labels(y_train, classes=10)

# --- set up network parameters
n_vis = X_train.shape[1]
n_out = T_train.shape[1]

encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask(img_shape).populate(encoders, rng=rng, flatten=True)

ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIF(),
    intercepts=nengo.dists.Choice([0.1]),
    max_rates=nengo.dists.Choice([100]),
    encoders=encoders,
    )

presentation_time = 0.1

with nengo.Network(seed=3) as model:
    model.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)

    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    a = nengo.Ensemble(n_hid, n_vis, **ens_params)
    v = nengo.Node(size_in=n_out)
    nengo.Connection(u, a, synapse=None)
    conn = nengo.Connection(
        a, v, eval_points=X_train, function=T_train, solver=solver)

    # --- image display
    image_shape = (1,) + img_shape
    display_f = image_display_function(image_shape, offset=1, scale=128)
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
    nengo.Connection(v, output.input)
