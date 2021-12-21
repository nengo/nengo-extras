import numpy as np


def net_diagram(net):
    """Create a .dot file showing nodes, ensmbles, and connections.

    This can be useful for debugging and testing builders that manipulate
    the model graph before construction.

    Parameters
    ----------
    net : Network
        A network from which objects and connections will be extracted.

    Returns
    -------
    text : string
        Text content of the desired .dot file.
    """
    objs = net.all_nodes + net.all_ensembles
    return obj_conn_diagram(objs, net.all_connections)


def obj_conn_diagram(objs, connections):
    """Create a .dot file showing nodes, ensmbles, and connections.

    This can be useful for debugging and testing builders that manipulate
    the model graph before construction.

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the nodes and ensembles in the model.
    connections : list of Connections
        All the connections in the model.

    Returns
    -------
    text : string
        Text content of the desired .dot file.
    """
    text = []
    text.append("digraph G {")
    for obj in objs:
        text.append('  "%d" [label="%s"];' % (id(obj), obj.label))

    def label(transform):
        # determine the label for a connection based on its transform
        transform = np.asarray(transform)
        if len(transform.shape) == 0:
            return ""
        return "%dx%d" % transform.shape

    for c in connections:
        text.append(
            '  "%d" -> "%d" [label="%s"];'
            % (id(c.pre_obj), id(c.post_obj), label(c.transform))
        )
    text.append("}")
    return "\n".join(text)
