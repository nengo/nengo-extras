import urllib

from nengo.utils.compat import pickle, PY2


urlretrieve = urllib.urlretrieve if PY2 else urllib.request.urlretrieve


if PY2:
    from cStringIO import StringIO  # noqa: F401

    def pickle_load(file, *args, **kwargs):
        return pickle.load(file, *args, **kwargs)

else:
    from io import StringIO  # noqa: F401

    def cmp(a, b):
        return (a > b) - (a < b)

    from pickle import _Unpickler
    class _UnpicklerStringsafe(_Unpickler):
        def _decode_string(self, value):
            if self.encoding == "bytes":
                return value
            else:
                try:
                    return value.decode(self.encoding, self.errors)
                except UnicodeDecodeError:
                    return value.decode('latin1')

    def pickle_load(file, *args, **kwargs):
        kwargs.setdefault('encoding', 'utf8')
        return _UnpicklerStringsafe(file, **kwargs).load()
