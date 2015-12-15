from collections import deque

import nengo
import numpy as np
from nengo.params import NdarrayParam, NumberParam


class PresentInput(nengo.processes.Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for `presentation_time` seconds.
    """
    inputs = NdarrayParam(shape=('*', '*'))
    presentation_time = NumberParam(low=0, low_open=True)

    def __init__(self, inputs, presentation_time):
        self.inputs = inputs
        self.presentation_time = presentation_time
        super(PresentInput, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == self.inputs[0].size

        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)

        def step_image_input(t):
            i = int(t / presentation_time + 1e-7)
            return inputs[i % n]

        return step_image_input


class DLBlackBox(nengo.processes.Process):
    def __init__(self, dl_net, history=1):
        self.dl_net = dl_net
        self.history = history

    def make_step(self, size_in, size_out, dt, rng):
        if size_in != self.dl_net.size_in:
            raise ValueError("DL model expects %d inputs (got %d)."
                             % (self.dl_net.size_in, size_in))
        if size_out != self.dl_net.size_out:
            raise ValueError("DL model produces %d outputs (got %d)."
                             % (self.dl_net.size_out, size_out))

        x_history = deque(maxlen=self.history)

        def step_dl_predict(t, x):
            # TODO this is pretty inefficient for Theano right now, as we
            # run the whole batch each time even though there's a lot of state
            # overlap. Not sure how to get around this at the moment...
            # probably will have to rewrite the updates.
            x_history.append(np.array(x, dtype=self.dl_net.dtype))
            x_in = np.asarray(x_history).reshape((len(x_history), 1, size_in))
            return self.dl_net._predict(x_in)[-1]

        return step_dl_predict
