import os

import nengo
from nengo.utils.compat import pickle
import numpy as np


def load_model_pickle(loadfile):
    loadfile = os.path.expanduser(loadfile)
    with open(loadfile, 'rb') as f:
        return pickle.load(f)


class CudaConvnetNetwork(nengo.Network):
    def __init__(self, model, synapse=None, lif_type='lif', **kwargs):
        super(CudaConvnetNetwork, self).__init__(**kwargs)
        self.model = model
        self.synapse = synapse
        self.lif_type = lif_type

        self.inputs = {}
        self.outputs = {}
        self.layer_outputs = {}

        with self:
            layers = model['model_state']['layers']
            for name in layers:
                self.add_layer(name, layers)

        if len(self.outputs) == 1:
            self.output = list(self.outputs.values())[0]

    def add_layer(self, name, layers):
        if name in self.layer_outputs:
            return

        layer = layers[name]
        for input_name in layer.get('inputs', []):
            if input_name not in self.layer_outputs:
                self.add_layer(input_name, layers)

        attrname = '_add_%s_layer' % layer['type'].replace('.', '_')
        if hasattr(self, attrname):
            output = getattr(self, attrname)(layer)
            self.layer_outputs[name] = output
        else:
            raise NotImplementedError(
                "Layer type %r not implemented" % layer['type'])

    def _get_inputs(self, layer):
        return [self.layer_outputs[i] for i in layer.get('inputs', [])]

    def _get_input(self, layer):
        assert len(layer.get('inputs', [])) == 1
        return self._get_inputs(layer)[0]

    def _add_data_layer(self, layer):
        node = nengo.Node(size_in=layer['outputs'], label=layer['name'])
        self.inputs[layer['name']] = node
        return node

    def _add_cost_logreg_layer(self, layer):
        labels, probs = self._get_inputs(layer)
        self.outputs[layer['name']] = probs
        return probs

    def _add_neuron_layer(self, layer):
        neuron = layer['neuron']
        ntype = neuron['type']
        n = layer['outputs']

        e = nengo.Ensemble(n, 1, label='%s_neurons' % layer['name'])
        e.gain = np.ones(n)
        e.bias = np.zeros(n)

        transform = 1.
        if ntype == 'ident':
            e.neuron_type = nengo.Direct()
        elif ntype == 'relu':
            e.neuron_type = nengo.RectifiedLinear()
        elif ntype == 'logistic':
            e.neuron_type = nengo.Sigmoid()
        elif ntype == 'softlif':
            from .neurons import SoftLIFRate
            tau_ref, tau_rc, alpha, amp, sigma, noise = [
                neuron['params'][k] for k in ['t', 'r', 'a', 'm', 'g', 'n']]
            lif_type = self.lif_type.lower()
            if lif_type == 'lif':
                e.neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
            elif lif_type == 'lifrate':
                e.neuron_type = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
            elif lif_type == 'softlifrate':
                e.neuron_type = SoftLIFRate(
                    sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            else:
                raise KeyError("Unrecognized LIF type %r" % self.lif_type)
            e.gain = alpha * np.ones(n)
            e.bias = np.ones(n)
            transform = amp
        else:
            raise NotImplementedError("Neuron type %r" % ntype)

        node = nengo.Node(size_in=n, label=layer['name'])
        nengo.Connection(self._get_input(layer), e.neurons, synapse=None)
        nengo.Connection(
            e.neurons, node, transform=transform, synapse=self.synapse)
        return node

    def _add_softmax_layer(self, layer):
        from .convnet import softmax
        node = nengo.Node(lambda t, x: softmax(x), size_in=layer['outputs'],
                          label=layer['name'])
        nengo.Connection(self._get_input(layer), node, synapse=None)
        return node

    def _add_dropout_layer(self, layer):
        node = nengo.Node(size_in=layer['outputs'], label=layer['name'])
        nengo.Connection(self._get_input(layer), node,
                         transform=layer['keep'], synapse=None)
        return node

    def _add_dropout2_layer(self, layer):
        return self._add_dropout_layer(layer)

    def _add_fc_layer(self, layer):
        pre = self._get_input(layer)
        weights = layer['weights'][0]
        biases = layer['biases'].ravel()
        node = nengo.Node(size_in=layer['outputs'], label=layer['name'])
        b = nengo.Node(output=biases, label='%s_biases' % layer['name'])
        nengo.Connection(pre, node, transform=weights.T, synapse=None)
        nengo.Connection(b, node, synapse=None)
        return node

    def _add_conv_layer(self, layer):
        from .convnet import Conv2d
        assert layer['sharedBiases']

        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]

        filters = layer['weights'][0].reshape(nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases']
        conv2d = Conv2d((nc, nx, nx), filters, biases, strides=st, padding=p)
        assert conv2d.shape_out == (nf, ny, ny)

        node = nengo.Node(conv2d, label=layer['name'])
        nengo.Connection(self._get_input(layer), node, synapse=None)
        return node

    def _add_local_layer(self, layer):
        from .convnet import Conv2d
        nc = layer['channels'][0]
        nx = layer['imgSize'][0]
        ny = layer['modulesX']
        nf = layer['filters']
        s = layer['filterSize'][0]
        st = layer['stride'][0]
        p = -layer['padding'][0]

        filters = layer['weights'][0].reshape(ny, ny, nc, s, s, nf)
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer['biases'][0].reshape(1, 1, 1)
        conv2d = Conv2d((nc, nx, nx), filters, biases, strides=st, padding=p)
        node = nengo.Node(conv2d, label=layer['name'])
        nengo.Connection(self._get_input(layer), node, synapse=None)
        return node

    def _add_pool_layer(self, layer):
        from .convnet import Pool2d
        assert layer['start'] == 0
        nc = layer['channels']
        nx = layer['imgSize']
        s = layer['sizeX']
        st = layer['stride']
        kind = layer['pool']

        pool2d = Pool2d((nc, nx, nx), s, strides=st, kind=kind, mode='full')
        node = nengo.Node(pool2d, label=layer['name'])
        nengo.Connection(self._get_input(layer), node, synapse=None)
        return node
