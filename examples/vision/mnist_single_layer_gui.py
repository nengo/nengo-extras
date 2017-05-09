"""
Single-layer NEF network applied to MNIST

Uses Nengo branch https://github.com/nengo/nengo/tree/function-points
"""
import nengo
import numpy as np

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from nengo_extras.gui import PresentImages


def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y


rng = np.random.RandomState(9)


# --- load the data
(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

train_targets = one_hot(y_train, 10)
test_targets = one_hot(y_test, 10)

# --- set up network parameters
n_vis = X_train.shape[1]
n_out = train_targets.shape[1]
# n_hid = 300
n_hid = 1000
# n_hid = 3000

# encoders = rng.normal(size=(n_hid, 11, 11))
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)

ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIFRate(),
    intercepts=nengo.dists.Choice([-0.5]),
    max_rates=nengo.dists.Choice([100]),
    encoders=encoders,
    )

solver = nengo.solvers.LstsqL2(reg=0.01)
# solver = nengo.solvers.LstsqL2(reg=0.0001)

presentation_time = 0.1

with nengo.Network(seed=3) as model:
    u = nengo.Node(PresentImages(X_test.reshape((-1, 1, 28, 28)),
                                 presentation_time))
    u.output.configure_display(offset=1., scale=128.)

    a = nengo.Ensemble(n_hid, n_vis, **ens_params)
    v = nengo.Node(size_in=n_out)
    nengo.Connection(u, a, synapse=None)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=train_targets, solver=solver)

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
