"""Export to GEXF for visualization of networks in Gephi."""

import weakref


class DispatchTable(object):
    """A descriptor to dispatch to other methods depending on argument type.

    How to use: assign the descriptor to a class attribute and use the
    `register` decorator to declare which functions to dispatch to for specific
    types::

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

    class InstDispatch(object):
        """Return value when accessing the dispatch table on an instance."""
        __slots__ = ('param', 'inst', 'owner')

        def __init__(self, param, inst, owner):
            self.param = param
            self.inst = inst
            self.owner = owner

        def __call__(self, obj):
            for cls in obj.__class__.__mro__:
                if cls in self.param.inst_type_table.get(self.inst, {}):
                    return self.param.inst_type_table[self.inst][cls](obj)
                elif cls in self.param.type_table:
                    return self.param.type_table[cls](self.inst, obj)
                elif self.param.parent is not None:
                    try:
                        return self.param.parent.__get__(
                            self.inst, self.owner)(obj)
                    except NotImplementedError:
                        pass
            raise NotImplementedError(
                "Nothing to dispatch to for type {}.".format(type(obj)))

        def register(self, type_, fn):
            if self.inst not in self.param.inst_type_table:
                self.param.inst_type_table[self.inst] = (
                    weakref.WeakKeyDictionary())
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
