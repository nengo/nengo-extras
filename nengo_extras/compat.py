import sys

import nengo

if nengo.version.version_info < (3, 0, 0):
    from nengo.utils.compat import is_integer, is_iterable
else:
    from nengo.utils.numpy import is_integer, is_iterable

PY2 = sys.version_info[0] == 2

if PY2:
    from cStringIO import StringIO
    from urllib import urlretrieve
    import Tkinter as tkinter
    import cPickle as pickle
else:
    from io import StringIO
    from urllib.request import urlretrieve
    import tkinter
    import pickle


def cmp(a, b):  # same as python2's builtin cmp, not available in python3
    return (a > b) - (a < b)


def pickle_load(file, *args, **kwargs):
    if not PY2:
        kwargs.setdefault('encoding', 'latin1')
    return pickle.load(file, *args, **kwargs)


def pickle_load_bytes(file, *args, **kwargs):
    if not PY2:
        kwargs.setdefault('encoding', 'bytes')
    return pickle.load(file, *args, **kwargs)
