#!/usr/bin/env python
import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    "version", os.path.join(root, "nengo_extras", "version.py"))
testing = "test" in sys.argv or "pytest" in sys.argv

install_requires = [
    "numpy>=1.8",
    "nengo",
]
plots_require = ["matplotlib>=1.4"]
deepnetworks_require = [
    "keras",
    "scipy",
    "theano",
]
docs_require = [
    "sphinx<1.7",
    "numpydoc>=0.6",
    "guzzle_sphinx_theme",
    "nbsphinx",
    "nbsphinx_link",
    "pillow",
    "jupyter",
    "matplotlib>=1.4",
] + deepnetworks_require + plots_require
tests_require = [
    "jupyter",
    "matplotlib>=1.4",
    "pytest>=3.2",
]


setup(
    name="nengo-extras",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/nengo/nengo-extras",
    license="Free for non-commercial use",
    description="Lesser used features for the Nengo neural simulator",
    long_description=read("README.rst", "CHANGES.rst"),
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=(["pytest-runner"] if testing else []) + ["numpy>=1.8"],
    install_requires=install_requires,
    extras_require={
        "deepnetworks": deepnetworks_require,
        "docs": docs_require,
        "plots": plots_require,
    },
    tests_require=tests_require,
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
