from __future__ import absolute_import

import collections
import os

import nengo
import numpy as np

from nengo_extras.compat import cmp, pickle_load
from nengo_extras.deepnetworks import SequentialNetwork
from nengo_extras.neurons import SoftLIFRate


def load_model_pickle(loadfile):
    loadfile = os.path.expanduser(loadfile)
    with open(loadfile, "rb") as f:
        return pickle_load(f)


def layer_depths(layers):
    depths = {}

    def get_depth(name):
        if name in depths:
            return depths[name]

        inputs = layers[name].get("inputs", [])
        depth = max(get_depth(i) for i in inputs) + 1 if len(inputs) > 0 else 0
        depths[name] = depth
        return depth

    for name in layers:
        get_depth(name)

    return depths


def sort_layers(layers, depths=None, cycle_check=True):
    depths = layer_depths(layers) if depths is None else depths

    def compare(a, b):
        da, db = depths[a], depths[b]
        return cmp(a, b) if da == db else cmp(da, db)

    snames = sorted(layers, cmp=compare)

    if cycle_check:

        def compare_cycle(a, b):
            ainb = a in layers[b].get("inputs", [])
            bina = b in layers[a].get("inputs", [])
            assert not (ainb and bina), "Cycle in graph"
            return -1 if ainb else 1 if bina else 0

        if any(
            compare_cycle(snames[i], snames[j]) < 0
            for i in range(1, len(snames))
            for j in range(i)
        ):
            raise ValueError("Cycle in graph")

    slayers = collections.OrderedDict((name, layers[name]) for name in snames)
    return slayers


class CudaConvnetNetwork(SequentialNetwork):
    def __init__(self, model, synapse=None, lif_type="lif", **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.synapse = synapse
        self.lif_type = lif_type

        # --- build model
        layers = dict(model["model_state"]["layers"])

        # remove cost layer(s) and label inputs
        for name, layer in list(layers.items()):
            if layer["type"].startswith("cost"):
                layers.pop(name)
                layers.pop(layer["inputs"][0])  # first input is labels

        for layer in layers.values():
            assert all(i in layers for i in layer.get("inputs", ()))

        # sort layers
        depths = layer_depths(layers)
        assert np.unique(list(depths.values())).size == len(depths)

        for name in sorted(layers, key=lambda n: depths[n]):
            self._add_layer(layers[name])

    def _add_layer(self, layer):
        attrname = "_add_%s_layer" % layer["type"].replace(".", "_")
        if hasattr(self, attrname):
            return getattr(self, attrname)(layer)
        else:
            raise NotImplementedError("Layer type %r not implemented" % layer["type"])

    def _get_inputs(self, layer):
        return [self.layers_by_name[i] for i in layer.get("inputs", [])]

    def _get_input(self, layer):
        assert len(layer.get("inputs", [])) == 1
        return self._get_inputs(layer)[0]

    def _add_data_layer(self, layer):
        d = layer["outputs"]
        return self.add_data_layer(d, name=layer["name"])

    def _add_neuron_layer(self, layer):
        inputs = [self._get_input(layer)]
        neuron = layer["neuron"]
        ntype = neuron["type"]
        n = layer["outputs"]

        gain = 1.0
        bias = 0.0
        amplitude = 1.0
        if ntype == "ident":
            neuron_type = nengo.Direct()
        elif ntype == "relu":
            neuron_type = nengo.RectifiedLinear()
        elif ntype == "logistic":
            neuron_type = nengo.Sigmoid()
        elif ntype == "softlif":

            tau_ref, tau_rc, alpha, amp, sigma = [
                neuron["params"][k] for k in ["t", "r", "a", "m", "g"]
            ]
            lif_type = self.lif_type.lower()
            if lif_type == "lif":
                neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
            elif lif_type == "lifrate":
                neuron_type = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
            elif lif_type == "softlifrate":
                neuron_type = SoftLIFRate(sigma=sigma, tau_rc=tau_rc, tau_ref=tau_ref)
            else:
                raise KeyError("Unrecognized LIF type %r" % self.lif_type)
            gain = alpha
            bias = 1.0
            amplitude = amp
        else:
            raise NotImplementedError("Neuron type %r" % ntype)

        return self.add_neuron_layer(
            n,
            inputs=inputs,
            neuron_type=neuron_type,
            synapse=self.synapse,
            gain=gain,
            bias=bias,
            amplitude=amplitude,
            name=layer["name"],
        )

    def _add_softmax_layer(self, layer):
        return None  # non-neural, we can do without it
        # inputs = [self._get_input(layer)]
        # return self.add_softmax_layer(
        #     d=layer['outputs'], inputs=inputs, name=layer['name'])

    def _add_dropout_layer(self, layer):
        inputs = [self._get_input(layer)]
        d = layer["outputs"]
        keep = layer["keep"]
        return self.add_dropout_layer(d, keep, inputs=inputs, name=layer["name"])

    def _add_dropout2_layer(self, layer):
        return self._add_dropout_layer(layer)

    def _add_fc_layer(self, layer):
        inputs = [self._get_input(layer)]
        weights = layer["weights"][0].T
        biases = layer["biases"].ravel()
        return self.add_full_layer(weights, biases, inputs=inputs, name=layer["name"])

    def _add_conv_layer(self, layer):
        inputs = [self._get_input(layer)]
        assert layer["sharedBiases"]
        nc = layer["channels"][0]
        nx = layer["imgSize"][0]
        ny = layer["modulesX"]
        nf = layer["filters"]
        s = layer["filterSize"][0]
        st = layer["stride"][0]
        p = -layer["padding"][0]

        filters = layer["weights"][0].reshape((nc, s, s, nf))
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer["biases"]
        layer = self.add_conv_layer(
            (nc, nx, nx),
            filters,
            biases,
            strides=st,
            padding=p,
            border="ceil",
            inputs=inputs,
            name=layer["name"],
        )
        assert layer.node.output.shape_out == (nf, ny, ny)
        return layer

    def _add_local_layer(self, layer):
        inputs = [self._get_input(layer)]
        nc = layer["channels"][0]
        nx = layer["imgSize"][0]
        ny = layer["modulesX"]
        nf = layer["filters"]
        s = layer["filterSize"][0]
        st = layer["stride"][0]
        p = -layer["padding"][0]

        filters = layer["weights"][0].reshape((ny, ny, nc, s, s, nf))
        filters = np.rollaxis(filters, axis=-1, start=0)
        biases = layer["biases"].reshape((nf, ny, ny))
        return self.add_local_layer(
            (nc, nx, nx),
            filters,
            biases,
            strides=st,
            padding=p,
            inputs=inputs,
            name=layer["name"],
        )

    def _add_pool_layer(self, layer):
        inputs = [self._get_input(layer)]
        assert layer["start"] == 0
        nc = layer["channels"]
        nx = layer["imgSize"]
        s = layer["sizeX"]
        st = layer["stride"]
        kind = layer["pool"]
        return self.add_pool_layer(
            (nc, nx, nx),
            s,
            strides=st,
            kind=kind,
            mode="full",
            inputs=inputs,
            name=layer["name"],
        )
