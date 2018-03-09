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
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_extras', 'version.py'))
testing = 'test' in sys.argv or 'pytest' in sys.argv

setup(
    name="nengo_extras",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/nengo/nengo_extras",
    license="Free for non-commercial use",
    description="Lesser used features for the Nengo neural simulator",
    long_description=read('README.rst', 'CHANGES.rst'),
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=["pytest-runner"] if testing else [] + [
        "numpy>=1.8",
    ],
    install_requires=[
        "nengo",
        "numpy>=1.8",
    ],
    extras_require={
        'deepnetworks': ["keras", "theano"],
        'plots': ["matplotlib"],
    },
    tests_require=[
        'pytest>=3.2',
    ],
)
