************
Nengo extras
************

Extra utilities and add-ons for Nengo.

This repository contains utilities that occupy
a liminal space not quite generic enough for inclusion in Nengo_,
but useful enough that they should be publicly accessible.

Some of these utilities may eventually migrate to Nengo_,
and others may be split off into their own separate repositories.

.. _Nengo: https://github.com/nengo/nengo

Installation
============

To install Nengo extras, we recommend using ``pip``.

.. code:: bash

   pip install nengo-extras

Usage
=====

Example notebooks can be found
in the ``docs/examples`` directory.

For a listing of the contents of this repository,
and information on how to use it,
see the `full documentation <https://www.nengo.ai/nengo-extras>`_.

Development
===========

To run the unit tests:

.. code-block:: bash

   pytest nengo_extras [--plots]

To run the static checks:

.. code-block:: bash

   .ci/static.sh run

To build the documentation:

.. code-block:: bash

   sphinx-build docs docs/_build
