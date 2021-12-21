import nengo

from . import reqs
from .version import version as __version__


def set_rc_default(rc, section, option, value=None):
    if not rc.has_section(section):
        rc.add_section(section)
    if not rc.has_option(section, option):
        rc.set(section, option, value=value)


set_rc_default(nengo.rc, "nengo_extras", "data_dir", ".")


__copyright__ = "2015-2021, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
