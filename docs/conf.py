# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import sys

try:
    import nengo_extras
    import guzzle_sphinx_theme
except ImportError:
    print("To build the documentation, nengo_extras and guzzle_sphinx_theme "
          "must be installed in the current environment. Please install these "
          "and their requirements first. A virtualenv is recommended!")
    sys.exit(1)


needs_sphinx = '1.3'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'guzzle_sphinx_theme',
    'numpydoc',
]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
}

# -- sphinx.ext.todo
todo_include_todos = True
# -- numpydoc config
numpydoc_show_class_members = False

# -- sphinx
exclude_patterns = ['_build']
source_suffix = ['.rst']
source_encoding = 'utf-8'
master_doc = 'index'


project = u'Nengo extras'
authors = u'Applied Brain Research'
# copyright = nengo_extras.__copyright__
version = '.'.join(
    nengo_extras.__version__.split('.')[:2])  # Short X.Y version
release = nengo_extras.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

pygments_style = "sphinx"
templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme"

html_theme_options = {
    "project_nav_name": "Nengo extras %s" % (version,),
    "base_url": "https://www.nengo.ai/nengo_extras",
}

html_title = "Nengo core {0} docs".format(release)
htmlhelp_basename = 'Nengo core'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo_extras.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo_extras', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo_extras', html_title, authors, 'Nengo extras',
     'Extra utilities and add-ons for Nengo.', 'Miscellaneous'),
]
