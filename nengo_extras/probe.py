import nengo

from nengo_extras.compat import is_iterable


def probe_all(net, recursive=False, probe_options=None, **probe_args):  # noqa: C901
    """Probes all objects in a network.

    Parameters
    ----------
    net : nengo.Network
    recursive : bool, optional (Default: False)
        Probe subnetworks recursively.
    probe_options: dict, optional (Default: None)
        A dict of the form {nengo_object_class: [attributes_to_probe]}.
        If None, every probeable attribute of every object will be probed.

    Returns
    -------
    A dictionary that maps objects and their attributes to their probes.

    Examples
    --------

    Probe the decoded output and spikes in all ensembles in a network and
    its subnetworks::

        with nengo.Network() as model:
            ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node1 = nengo.Node(output=[0])
            conn = nengo.Connection(node1, ens1)
            subnet = nengo.Network(label='subnet')
            with subnet:
                ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
                node2 = nengo.Node(output=[0])
        probe_options = {nengo.Ensemble: ['decoded_output', 'spikes']}
        probes = probe_all(model, recursive=True, probe_options=probe_options)

    """

    probes = {}

    def all_probes(obj):
        if probe_options is not None and type(obj) in probe_options:
            attrs = probe_options[type(obj)]
        else:
            attrs = obj.probeable

        return {attr: nengo.Probe(obj, attr, **probe_args) for attr in attrs}

    ensembles = net.all_ensembles if recursive else net.ensembles
    nodes = net.all_nodes if recursive else net.nodes
    connections = net.all_connections if recursive else net.connections

    with net:
        if probe_options is None or nengo.Ensemble in probe_options:
            for ens in ensembles:
                probes[ens] = all_probes(ens)

        if probe_options is None or nengo.ensemble.Neurons in probe_options:
            for ens in ensembles:
                probes[ens.neurons] = all_probes(ens.neurons)

        if probe_options is None or nengo.Node in probe_options:
            for node in nodes:
                probes[node] = all_probes(node)

        if probe_options is None or nengo.Connection in probe_options:
            for conn in connections:
                probes[conn] = all_probes(conn)

        LearningRule = nengo.connection.LearningRule
        if probe_options is None or LearningRule in probe_options:
            for conn in connections:
                lr = conn.learning_rule
                if lr is None:
                    continue
                if not is_iterable(lr):
                    lr = [lr]
                for rule in lr:
                    probes[rule] = all_probes(rule)

    return probes
