from .version import version as __version__
from .rc import rc

# --- nengo_extras namespace (API)
from .convnet import Conv2d, Pool2d
from .neurons import FastLIF, SoftLIFRate
from . import (
    camera, data, dists, graphviz, gui, networks, neurons, probe, vision)

__copyright__ = "2015-2018, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
