"""
Use Nengo's RC files to manage our RC parameters.
"""
import types

from nengo import rc


def rc_setdefault(self, section, option, value=None):
    if not self.has_section(section):
        self.add_section(section)
    if not self.has_option(section, option):
        self.set(section, option, value=value)


rc.setdefault = types.MethodType(rc_setdefault, rc)

# --- set our defaults
rc.setdefault('nengo_extras', 'data_dir', '.')
