import pytest

from nengo_extras.gexf import DispatchTable


def test_can_dispatch_table_defaults():
    class A(object):
        def __init__(self):
            self.called = False

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = True

    a = A()
    Test().dispatch(a)
    assert a.called


def test_dispatch_obj_inheritance():
    class A(object):
        def __init__(self):
            self.called = False

    class B(A):
        pass

    class C(A):
        pass

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'a'

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'b'

    a = A()
    Test().dispatch(a)
    assert a.called == 'a'

    b = B()
    Test().dispatch(b)
    assert b.called == 'b'

    c = C()
    Test().dispatch(c)
    assert c.called == 'a'


def test_dispatch_cls_inheritance():
    class A(object):
        def __init__(self):
            self.called = False

    class B(object):
        def __init__(self):
            self.called = False

    class C(object):
        def __init__(self):
            self.called = False

    class Test1(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'test1.process_a'

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'test1.process_b'

    class Test2(Test1):
        dispatch = DispatchTable(Test1.dispatch)

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'test2.process_b'

        @dispatch.register(C)
        def process_c(self, obj):
            obj.called = 'test2.process_c'

    a = A()
    b = B()
    Test1().dispatch(a)
    assert a.called == 'test1.process_a'
    Test1().dispatch(b)
    assert b.called == 'test1.process_b'

    a = A()
    b = B()
    c = C()
    Test2().dispatch(a)
    assert a.called == 'test1.process_a'
    Test2().dispatch(b)
    assert b.called == 'test2.process_b'
    Test2().dispatch(c)
    assert c.called == 'test2.process_c'


def test_dispatch_instance_specific():
    class A(object):
        def __init__(self):
            self.called = False

    class B(object):
        def __init__(self):
            self.called = False

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'test.process_a'

    def proc_inst(obj):
        obj.called = 'proc_inst'

    test = Test()
    test2 = Test()
    test.dispatch.register(A, proc_inst)
    test.dispatch.register(B, proc_inst)

    a = A()
    test.dispatch(a)
    assert a.called == 'proc_inst'
    test2.dispatch(a)
    assert a.called == 'test.process_a'

    b = B()
    test.dispatch(b)
    assert b.called == 'proc_inst'


def test_dispatch_errors():
    class Test(object):
        dispatch = DispatchTable()

    with pytest.raises(NotImplementedError):
        Test().dispatch(object())
