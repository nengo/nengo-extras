from nengo.utils.compat import pickle, PY2

if PY2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


if PY2:
    from cStringIO import StringIO  # noqa: F401
else:
    from io import StringIO  # noqa: F401

    def cmp(a, b):
        return (a > b) - (a < b)


def pickle_load(file, *args, **kwargs):
    if not PY2:
        kwargs.setdefault('encoding', 'latin1')
    return pickle.load(file, *args, **kwargs)


def pickle_load_bytes(file, *args, **kwargs):
    if not PY2:
        kwargs.setdefault('encoding', 'bytes')
    return pickle.load(file, *args, **kwargs)
