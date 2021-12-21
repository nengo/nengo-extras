"""Export to GEXF for visualization of networks in Gephi."""

import weakref
import xml.etree.ElementTree as et
from collections import OrderedDict, namedtuple
from collections.abc import Mapping, Sequence
from datetime import date

import nengo

try:
    import nengo_spa as spa
except ImportError:
    spa = None
import numpy as np


class DispatchTable:
    """A descriptor to dispatch to other methods depending on argument type.

    How to use: assign the descriptor to a class attribute and use the
    ``register`` decorator to declare which functions to dispatch to
    for specific types::

        class MyClass(object):
            dispatch = DispatchTable()

            @dispatch.register(TypeA)
            def handle_type_a(self, obj_of_type_a):
                # ...

            @dispatch.register(TypeB)
            def handle_type_b(self, obj_of_type_b):
                # ...

    To then call the method for the appropriate type::

        inst = MyClass()
        inst.dispatch(obj_of_type_a_or_b)

    If multiple methods would match (e.g. if *TypeB* inherits from *TypeA*),
    the most specific method will be used (to be precise: the first type in the
    method resolution order with a registered method will be used).

    The *DispatchTable* descriptor accepts another *DispatchTable* as argument
    which will be used as a fallback. This allows to inherit the dispatch
    table and selectively overwrite methods like so::

        class Inherited(MyClass):
            dispatch = DispatchTable(MyClass.dispatch)

            @dispatch.register(TypeA)
            def alternate_type_a_handler(self, obj_of_type_a):
                # ...

    Finally, dispatch methods can also be changed on a per-instance basis::

        inst.dispatch.register(TypeA, inst_type_a_handler)
    """

    class InstDispatch:
        """Return value when accessing the dispatch table on an instance."""

        __slots__ = ("param", "inst", "owner")

        def __init__(self, param, inst, owner):
            self.param = param
            self.inst = inst
            self.owner = owner

        def __call__(self, obj):
            for cls in obj.__class__.__mro__:
                if cls in self.param.inst_type_table.get(self.inst, {}):
                    return self.param.inst_type_table[self.inst][cls](obj)
                if cls in self.param.type_table:
                    return self.param.type_table[cls](self.inst, obj)
                if self.param.parent is not None:
                    try:
                        return self.param.parent.__get__(self.inst, self.owner)(obj)
                    except NotImplementedError:
                        pass
            raise NotImplementedError(
                "Nothing to dispatch to for type {}.".format(type(obj))
            )

        def register(self, type_, fn):
            if self.inst not in self.param.inst_type_table:
                self.param.inst_type_table[self.inst] = weakref.WeakKeyDictionary()
            table = self.param.inst_type_table[self.inst]
            table[type_] = fn
            return fn

    def __init__(self, parent=None):
        self.type_table = weakref.WeakKeyDictionary()
        self.inst_type_table = weakref.WeakKeyDictionary()
        self.parent = parent

    def register(self, type_):
        def _register(fn):
            assert type_ not in self.type_table
            self.type_table[type_] = fn
            return fn

        return _register

    def __get__(self, inst, owner):
        if inst is None:
            return self
        return self.InstDispatch(self, inst, owner)


class HierarchicalLabeler:
    """Obtains labels for objects in a Nengo network.

    The names will include the network hierarchy.

    Usage example::

        labels = HierarchicalLabeler().get_labels(model)
    """

    dispatch = DispatchTable()

    def __init__(self):
        self._names = None

    @dispatch.register(Sequence)
    def get_labels_from_sequence(self, seq):
        base_name = self._names[seq]
        for i, obj in enumerate(seq):
            self._handle_found_name(
                obj, "{base_name}[{i}]".format(base_name=base_name, i=i)
            )

    @dispatch.register(Mapping)
    def get_labels_from_mapping(self, mapping):
        base_name = self._names[mapping]
        for k in mapping:
            obj = mapping[k]
            self._handle_found_name(
                obj, "{base_name}[{k}]".format(base_name=base_name, k=k)
            )

    @dispatch.register(object)
    def get_labels_from_object(self, obj):
        pass

    @dispatch.register(nengo.Network)
    def get_labels_from_network(self, net):
        if net in self._names:
            base_name = self._names[net] + "."
        else:
            base_name = ""

        check_last = {"ensembles", "nodes", "connections", "networks", "probes"}
        check_never = {
            "all_ensembles",
            "all_nodes",
            "all_connections",
            "all_networks",
            "all_objects",
            "all_probes",
        }

        for name in dir(net):
            if not name.startswith("_") and name not in check_last | check_never:
                try:
                    attr = getattr(net, name)
                except AttributeError:
                    pass
                else:
                    self._handle_found_name(attr, base_name + name)

        for name in check_last:
            attr = getattr(net, name)
            self._handle_found_name(attr, base_name + name)

    def _handle_found_name(self, obj, name):
        if (
            isinstance(obj, (nengo.base.NengoObject, nengo.Network))
            and obj not in self._names
        ):
            self._names[obj] = name
            self.dispatch(obj)

    def get_labels(self, model):
        self._names = weakref.WeakKeyDictionary()
        self.dispatch(model)
        return self._names


Attr = namedtuple("Attr", ["id", "type", "default"])


class GexfConverter:
    """Converts Nengo models into GEXF files.

    This can be loaded in Gephi for visualization of the model graph.

    Links:

    * `Gephi <https://gephi.org/>`_
    * `GEXF <https://github.com/gephi/gexf/wiki>`_

    This class can be inherited from to customize the conversion or
    alternatively the ``dispatch`` table can be changed on a
    per-instance basis.

    Note that probes are currently not included in the graph.

    The following attributes will be stored on graph nodes:

    * *type*: type of the Nengo object (e.g., *nengo.ensemble.Ensemble*),
    * *net*: unique ID of the containing network,
    * *net_label*: (possibly non-unique) label of the containing network,
    * *size_in*: input size,
    * *size_out*: output_size,
    * *radius*: ensemble radius (unset for other nodes),
    * *n_neurons*: number of neurons (0 for non-ensembles),
    * *neuron_type*: string representation of the neuron type (unset for
      non-ensembles).

    The following attributes will be stored on graph edges:

    * *pre_type*: type of the connection's pre object (e.g.,
      *nengo.ensemble.Neurons*),
    * *post_type*: type of the connection's post object (e.g.,
      *nengo.ensemble.Neurons*),
    * *synapse*: string representation of the synapse,
    * *tau*: the tau parameter of the synapse if existent,
    * *function*: string representation of the connection's function,
    * *transform*: string representation of the connection's transform,
    * *scalar_transform*: float representation of the transform if it is a
      scalar,
    * *learning_rule_type*: string representation of the connection's learning
      rule type.

    Parameters
    ----------
    labeler : optional
        Object with a ``get_labels`` method that returns a dictionary mapping
        model objects to labels. If not given, a new `HierarchicalLabeler`
        will be used.
    hierarchical : bool, optional (default: False)
        Whether to include information of the network hierarchy in the file.
        Support for hierarchical graphs was removed in Gephi 0.9 and
        hierarchical networks will be automatically flattened which leaves an
        unconnected node for every network.

    Examples
    --------

    Basic usage to write a GEXF file::

        GexfConverter().convert(model).write('model.gexf')
    """

    dispatch = DispatchTable()

    node_attrs = OrderedDict(
        (
            ("type", Attr(0, "string", None)),
            ("net", Attr(1, "long", None)),
            ("net_label", Attr(2, "string", None)),
            ("size_in", Attr(3, "integer", None)),
            ("size_out", Attr(4, "integer", None)),
            ("radius", Attr(5, "float", None)),
            ("n_neurons", Attr(6, "integer", 0)),
            ("neuron_type", Attr(7, "string", None)),
        )
    )
    edge_attrs = OrderedDict(
        (
            ("pre_type", Attr(0, "string", None)),
            ("post_type", Attr(1, "string", None)),
            ("synapse", Attr(2, "string", None)),
            ("tau", Attr(3, "float", None)),
            ("function", Attr(4, "string", None)),
            ("transform", Attr(5, "string", None)),
            ("scalar_transform", Attr(6, "float", 1.0)),
            ("learning_rule_type", Attr(7, "string", None)),
        )
    )

    def __init__(self, labeler=None, hierarchical=False):
        if labeler is None:
            labeler = HierarchicalLabeler()
        self.labeler = labeler
        self.hierarchical = hierarchical
        self.version = (1, 3)
        self.tag = "draft"

        # State used during processing of a model
        # WeakKeyDict so we don't prevent garbage collection after conversion
        # finished.
        self._labels = weakref.WeakKeyDictionary()
        self._net = None

    def convert(self, model):
        """Convert a model to GEXF format.

        Returns
        -------
        xml.etree.ElementTree.ElementTree
            Converted model.
        """
        self._labels = self.labeler.get_labels(model)
        self._labels[model] = "model"
        return self.make_document(model)

    def make_document(self, model):
        """Create the GEXF XML document from *model*.

        This method is exposed so it can be overwritten in inheriting classes.
        Invoke `convert` instead of this method to convert a model.

        Returns
        -------
        xml.etree.ElementTree.ElementTree
            Converted model.
        """
        version = ".".join(str(i) for i in self.version)
        tag_version = version + self.tag
        gexf = et.Element(
            "gexf",
            {
                "xmlns": "http://www.gexf.net/" + tag_version,
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": (
                    "http://www.gexf.net/"
                    + tag_version
                    + " "
                    + "http://www.gexf.net/"
                    + tag_version
                    + "/gexf.xsd"
                ),
                "version": version,
            },
        )

        meta = et.SubElement(
            gexf, "meta", {"lastmodifieddate": date.today().isoformat()}
        )
        creator = et.SubElement(meta, "creator")
        creator.text = self.get_typename(self)

        graph = et.SubElement(gexf, "graph", {"defaultedgetype": "directed"})
        graph.append(self.make_attr_defs("node", self.node_attrs))
        graph.append(self.make_attr_defs("edge", self.edge_attrs))

        graph.append(self.dispatch(model))

        edges = et.SubElement(graph, "edges")
        for c in model.all_connections:
            elem = self.dispatch(c)
            if elem is not None:
                edges.append(elem)

        return et.ElementTree(gexf)

    def make_attr_defs(self, cls, defs):
        """Generate an attribute definition block.

        Parameters
        ----------
        cls : str
            Class the attribute definitions are for ('node' or 'edge').
        defs : dict
            Attribute definitions. Maps attribute names to `Attr` instances.

        Returns
        -------
        xml.etree.ElementTree.Element
        """
        attributes = et.Element("attributes", {"class": cls})
        for k, d in defs.items():
            attr = et.SubElement(
                attributes,
                "attribute",
                {
                    "id": str(d.id),
                    "title": k,
                    "type": d.type,
                },
            )
            if d.default is not None:
                default = et.SubElement(attr, "default")
                default.text = str(d.default)
        return attributes

    def make_attrs(self, defs, attrs):
        """Generates a block of attribute values.

        Parameters
        ----------
        defs : dict
            Attribute definitions. Maps attribute names to `Attr` instances.
        attrs : dict
            Mapping of attribute names to assigned values.

        Returns
        -------
        xml.etree.ElementTree.Element
        """
        values = et.Element("attvalues")
        assert all(k in defs for k in attrs.keys())
        for k, d in defs.items():
            if k in attrs and attrs[k] is not None:
                values.append(
                    et.Element(
                        "attvalue",
                        {
                            "for": str(d.id),
                            "value": str(attrs[k]),
                        },
                    )
                )
        return values

    def make_node(self, obj, **attrs):
        """Generate a node for *obj* with attributes *attrs*."""
        tag_attrib = {"id": str(id(obj))}
        if obj in self._labels:
            tag_attrib["label"] = self._labels[obj]
        node = et.Element("node", tag_attrib)
        if len(attrs) > 0:
            node.append(self.make_attrs(self.node_attrs, attrs))
        return node

    def make_edge(self, obj, source, target, **attrs):
        "Edge for *obj* from *source* to *target* with attributes *attrs*."
        tag_attrib = {
            "id": str(id(obj)),
            "source": str(id(source)),
            "target": str(id(target)),
        }
        edge = et.Element("edge", tag_attrib)
        if len(attrs) > 0:
            edge.append(self.make_attrs(self.edge_attrs, attrs))
        return edge

    @dispatch.register(nengo.Network)
    def convert_network(self, net):
        parent_net = self._net
        self._net = net

        nodes = et.Element("nodes")
        leaves = net.ensembles + net.nodes + net.probes
        for leave in leaves:
            leave_elem = self.dispatch(leave)
            if leave_elem is not None:
                nodes.append(leave_elem)
        if self.hierarchical:
            for subnet in net.networks:
                subnet_node = self.make_node(
                    subnet,
                    type=self.get_typename(subnet),
                    net=id(self._net),
                    net_label=self._labels.get(self._net, None),
                    n_neurons=subnet.n_neurons,
                )
                subnet_node.append(self.dispatch(subnet))
                nodes.append(subnet_node)
        else:
            for subnet in net.networks:
                nodes.extend(self.dispatch(subnet))

        self._net = parent_net
        return nodes

    @dispatch.register(nengo.Ensemble)
    def convert_ensemble(self, ens):
        return self.make_node(
            ens,
            type=self.get_typename(ens),
            net=id(self._net),
            net_label=self._labels.get(self._net, None),
            size_in=ens.dimensions,
            size_out=ens.dimensions,
            radius=ens.radius,
            n_neurons=ens.n_neurons,
            neuron_type=ens.neuron_type,
        )

    @dispatch.register(nengo.Node)
    def convert_node(self, node):
        return self.make_node(
            node,
            type=self.get_typename(node),
            net=id(self._net),
            net_label=self._labels.get(self._net, None),
            size_in=node.size_in,
            size_out=node.size_out,
        )

    @dispatch.register(nengo.Probe)
    def convert_probe(self, probe):
        return None

    @dispatch.register(nengo.Connection)
    def convert_connection(self, conn):
        source = self.get_node_obj(conn.pre_obj)
        target = self.get_node_obj(conn.post_obj)
        return self.make_edge(
            conn,
            source,
            target,
            pre_type=self.get_typename(conn.pre_obj),
            post_type=self.get_typename(conn.post_obj),
            synapse=conn.synapse,
            tau=conn.synapse.tau if hasattr(conn.synapse, "tau") else None,
            function=conn.function,
            transform=conn.transform,
            scalar_transform=(conn.transform if np.isscalar(conn.transform) else None),
            learning_rule_type=conn.learning_rule_type,
        )

    def get_node_obj(self, obj):
        """Get an object with a corresponding graph node related to *obj*.

        For certain objects like `nengo.ensemble.Neurons` or
        `nengo.connection.LearningRule` no graph node will be created. This
        function will resolve such an object to a related object that has a
        corresponding graph node (e.g., the ensemble for a neurons object or
        the pre object for a learning rule).

        In `GexfConverter` this is used to make sure connections are between
        the correct nodes and do not introduce unrelated dangling nodes.
        """
        if isinstance(obj, nengo.ensemble.Neurons):
            return obj.ensemble
        if isinstance(obj, nengo.connection.LearningRule):
            return self.get_node_obj(obj.connection.pre_obj)
        return obj

    @classmethod
    def get_typename(cls, obj):
        tp = type(obj)
        return tp.__module__ + "." + tp.__name__


class CollapsingGexfConverter(GexfConverter):
    """Converts Nengo models into GEXF files with some collapsed networks.

    See `GexfConverter` for general information on conversion to GEXF files.
    This class will collapse certain networks to a single node in the
    conversion.

    Parameters
    ----------
    to_collapse : sequence, optional
        Network types to collapse, if not given the networks listed in
        ``NENGO_NETS`` and ``SPA_NETS`` will be collapsed. Note that
        ``SPA_NETS`` currently only contains networks from *nengo_spa*,
        but not the *spa* module in core *nengo*.
    labeler : optional
        Object with a ``get_labels`` method that returns a dictionary mapping
        model objects to labels. If not given, a new `HierarchicalLabeler`
        will be used.
    hierarchical : bool, optional (default: False)
        Whether to include information of the network hierarchy in the file.
        Support for hierarchical graphs was removed in Gephi 0.9 and
        hierarchical networks will be automatically flattened which leaves an
        unconnected node for every network.
    """

    dispatch = DispatchTable(GexfConverter.dispatch)

    NENGO_NETS = (
        nengo.networks.CircularConvolution,
        nengo.networks.EnsembleArray,
        nengo.networks.Product,
    )
    if spa is None:
        SPA_NETS = ()
    else:
        SPA_NETS = (
            spa.networks.CircularConvolution,
            spa.AssociativeMemory,
            spa.Bind,
            spa.Compare,
            spa.Product,
            spa.Scalar,
            spa.State,
            spa.Transcode,
        )

    def __init__(self, to_collapse=None, labeler=None, hierarchical=False):
        super().__init__(labeler=labeler, hierarchical=hierarchical)

        if to_collapse is None:
            to_collapse = self.NENGO_NETS + self.SPA_NETS

        for cls in to_collapse:
            self.dispatch.register(cls, self.convert_collapsed)

        self.obj2collapsed = weakref.WeakKeyDictionary()

    def convert_collapsed(self, net):
        """Used to convert a network into a collapsed graph node."""
        nodes = et.Element("nodes")
        nodes.append(
            self.make_node(
                net,
                type=self.get_typename(net),
                net=id(self._net),
                net_label=self._labels.get(self._net, None),
                n_neurons=net.n_neurons,
            )
        )
        self.obj2collapsed.update({child: net for child in net.all_objects})
        return nodes

    def get_node_obj(self, obj):
        obj = super().get_node_obj(obj)
        return self.obj2collapsed.get(obj, obj)
