# pylint: disable=broad-except,unused-import

import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow

    HAS_TENSORFLOW = True
except Exception as err:
    HAS_TENSORFLOW = False
    logger.debug("Error importing TensorFlow:\n%s", err)

try:
    import nengo_dl.neuron_builders

    HAS_NENGO_DL = True
    assert HAS_TENSORFLOW, "NengoDL installed without Tensorflow"
except Exception as err:
    HAS_NENGO_DL = False
    logger.debug("Error importing NengoDL:\n%s", err)


try:
    import nengo_loihi

    HAS_NENGO_LOIHI = True
except Exception as err:
    HAS_NENGO_LOIHI = False
    logger.debug("Error importing NengoLoihi:\n%s", err)

try:
    import gi

    gi.require_version("Gst", "1.0")
    import gi.repository

    HAS_GI = True
except Exception as err:
    HAS_GI = False
    logger.debug("Error importing PyGObject:\n%s", err)


try:
    import scipy.cluster.hierarchy
    import scipy.interpolate
    import scipy.ndimage.interpolation
    import scipy.stats

    HAS_SCIPY = True
except Exception as err:
    HAS_SCIPY = False
    logger.debug("Error importing SciPy:\n%s", err)


try:
    import keras.backend
    import keras.layers

    HAS_KERAS = True
except Exception as err:
    HAS_KERAS = False
    logger.debug("Error importing Keras:\n%s", err)


try:
    import numba.extending

    HAS_NUMBA = True
except Exception as err:
    HAS_NUMBA = False
    logger.debug("Error importing Numba:\n%s", err)
