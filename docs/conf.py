# -*- coding: utf-8 -*-
#
# Automatically generated by nengo-bones, do not edit this file directly

import pathlib

import nengo_extras

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "nbsphinx",
    "nengo_sphinx_theme",
    "nengo_sphinx_theme.ext.backoff",
    "nengo_sphinx_theme.ext.sourcelinks",
    "notfound.extension",
    "numpydoc",
    "nengo_sphinx_theme.ext.resolvedefaults",
]

# -- sphinx.ext.autodoc
autoclass_content = "both"  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.doctest
doctest_global_setup = """
import nengo_extras
import nengo
"""

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "nengo": ("https://www.nengo.ai/nengo/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- nbsphinx
nbsphinx_timeout = -1

# -- notfound.extension
notfound_template = "404.html"
notfound_urls_prefix = "/nengo-extras/"

# -- numpydoc config
numpydoc_show_class_members = False

# -- nengo_sphinx_theme.ext.sourcelinks
sourcelinks_module = "nengo_extras"
sourcelinks_url = "https://github.com/nengo/nengo-extras"

# -- sphinx
nitpicky = True
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]
linkcheck_timeout = 30
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
linkcheck_ignore = [r"http://localhost:\d+"]
linkcheck_anchors = True
default_role = "py:obj"
pygments_style = "sphinx"
user_agent = "nengo_extras"

project = "NengoExtras"
authors = "Applied Brain Research"
copyright = "2015-2022 Applied Brain Research"
version = ".".join(nengo_extras.__version__.split(".")[:2])  # Short X.Y version
release = nengo_extras.__version__  # Full version, with tags

# -- HTML output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "nengo_sphinx_theme"
html_title = f"NengoExtras {release} docs"
htmlhelp_basename = "NengoExtras"
html_last_updated_fmt = ""  # Default output format (suppressed)
html_show_sphinx = False
html_favicon = str(pathlib.Path("_static", "favicon.ico"))
html_theme_options = {
    "nengo_logo": "general-full-light.svg",
    "nengo_logo_color": "#a8acaf",
    "tagmanager_id": "GTM-KWCR2HN",
}
