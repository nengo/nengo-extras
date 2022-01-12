**************
Loihi training
**************

When training models for Loihi using NengoDL,
you can improve performance by better matching
the chip's neuron cores using the ``LoihiLIF``
and ``LoihiSpikingRectifiedLinear`` neuron types.
This module, which is automatically used by NengoLoihi,
adds builders to NengoDL that allow those neuron types
to build and train properly.

Neuron output noise models
==========================

.. autoclass:: nengo_extras.loihi_training.NeuronOutputNoise

.. autoclass:: nengo_extras.loihi_training.LowpassRCNoise

.. autoclass:: nengo_extras.loihi_training.AlphaRCNoise


NengoDL builders
================

.. autoclass:: nengo_extras.loihi_training.NoiseBuilder

.. autoclass:: nengo_extras.loihi_training.NoNoiseBuilder

.. autoclass:: nengo_extras.loihi_training.LowpassRCNoiseBuilder

.. autoclass:: nengo_extras.loihi_training.AlphaRCNoiseBuilder

.. autoclass:: nengo_extras.loihi_training.LoihiLIFBuilder

.. autoclass:: nengo_extras.loihi_training.LoihiSpikingRectifiedLinearBuilder
