***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

0.5.0 (January 12, 2022)
========================

**Added**

- Added NengoDL builders for NengoLoihi neuron types. These will automatically be used
  by the NengoLoihi repository if it is installed. (`#94`_)

**Fixed**

- Updated to work with more recent versions of Nengo. (`#94`_)

.. _#94: https://github.com/nengo/nengo/pull/94

0.4.0 (November 15, 2019)
=========================

**Added**

- Added ``nengo_extras.simulators.RealTimeSimulator``, which will ensure that
  simulations don't run faster than real time.
  (`#85 <https://github.com/nengo/nengo-extras/pull/85>`_,
  `#151 <https://github.com/nengo/nengo/pull/151>`_)
- Added ``nengo_extras.neurons.NumbaLIF``, which is a numba-accelerated
  drop-in replacement for the ``nengo.LIF`` neuron model (requires ``numba`` to
  be installed).
  (`#86 <https://github.com/nengo/nengo-extras/pull/86>`_)

**Fixed**

- Fixed some Nengo 3.0.0 compatibility issues.
  (`#90 <https://github.com/nengo/nengo-extras/pull/90>`_)

0.3.0 (June 4, 2018)
====================

**Changed**

- Submodules are no longer automatically imported into the
  ``nengo_extras`` namespace, as it can be difficult to install
  requirements for the various tools in Nengo Extras.
  (`#77 <https://github.com/nengo/nengo-extras/issues/77>`_,
  `#78 <https://github.com/nengo/nengo-extras/pull/78>`_)

0.2.0 (May 31, 2018)
====================

**Added**

- Added the association matrix learning rule (AML)
  to learn associations from cue vectors to target vectors
  in a one-shot fashion without catastrophic forgetting.
  (`#72 <https://github.com/nengo/nengo-extras/pull/72>`_)
- Added classes to convert Nengo models to GEXF for visualization with Gephi.
  (`#54 <https://github.com/nengo/nengo-extras/pull/54>`_)
- Added a ``Camera`` process to stream images from a camera to Nengo.
  (`#61 <https://github.com/nengo/nengo-extras/pull/61>`_)

0.1.0 (March 14, 2018)
======================

Initial release of Nengo Extras!
Tested with Nengo 2.7.0, but should work with earlier versions.
If you run into any issues, please
`file a bug report <https://github.com/nengo/nengo-extras/issues/new>`_.
