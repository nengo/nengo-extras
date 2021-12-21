*************
Deep learning
*************

Classes and utilities for doing deep learning with Nengo.

.. warning::

   These utilities were created before `Nengo DL`_,
   which provides tighter integration between
   Nengo and TensorFlow.
   If you are new to Nengo or new to doing deep learning in Nengo,
   we recommend that you first check out `Nengo DL`_
   to see if it fits your use case.

.. _Nengo DL: https://www.nengo.ai/nengo-dl

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/keras/mnist_spiking_cnn
   examples/cuda_convnet/cifar10_spiking_cnn
   examples/cuda_convnet/imagenet_spiking_cnn

Datasets
========

The following functions make use of Nengo's RC file
to set the location for data downloads.
See the `Nengo configuration <https://www.nengo.ai/nengo/nengorc.html>`_
documentation for instructions of editing RC files.
Configuration settings are listed below.

.. code:: bash

    [nengo_extras]
    # directory to store downloaded datasets
    data_dir = ~/data

.. autosummary::

   nengo_extras.data.get_cifar10_tar_gz
   nengo_extras.data.get_cifar100_tar_gz
   nengo_extras.data.get_ilsvrc2012_tar_gz
   nengo_extras.data.get_mnist_pkl_gz
   nengo_extras.data.get_svhn_tar_gz
   nengo_extras.data.load_cifar10
   nengo_extras.data.load_cifar100
   nengo_extras.data.load_ilsvrc2012
   nengo_extras.data.load_mnist
   nengo_extras.data.load_svhn
   nengo_extras.data.spasafe_name
   nengo_extras.data.spasafe_names
   nengo_extras.data.one_hot_from_labels
   nengo_extras.data.ZCAWhiten

.. autofunction:: nengo_extras.data.get_cifar10_tar_gz

.. autofunction:: nengo_extras.data.get_cifar100_tar_gz

.. autofunction:: nengo_extras.data.get_ilsvrc2012_tar_gz

.. autofunction:: nengo_extras.data.get_mnist_pkl_gz

.. autofunction:: nengo_extras.data.get_svhn_tar_gz

.. autofunction:: nengo_extras.data.load_cifar10

.. autofunction:: nengo_extras.data.load_cifar100

.. autofunction:: nengo_extras.data.load_ilsvrc2012

.. autofunction:: nengo_extras.data.load_mnist

.. autofunction:: nengo_extras.data.load_svhn

.. autofunction:: nengo_extras.data.spasafe_name

.. autofunction:: nengo_extras.data.spasafe_names

.. autofunction:: nengo_extras.data.one_hot_from_labels

.. autoclass:: nengo_extras.data.ZCAWhiten

Keras
=====

.. autosummary::

   nengo_extras.keras.SoftLIF
   nengo_extras.keras.load_model_pair
   nengo_extras.keras.save_model_pair
   nengo_extras.keras.LSUVinit

.. autoclass:: nengo_extras.keras.SoftLIF

.. autofunction:: nengo_extras.keras.load_model_pair

.. autofunction:: nengo_extras.keras.save_model_pair

.. autofunction:: nengo_extras.keras.LSUVinit

Networks
========

.. autosummary::

   nengo_extras.deepnetworks.DeepNetwork
   nengo_extras.deepnetworks.SequentialNetwork
   nengo_extras.keras.SequentialNetwork
   nengo_extras.deepnetworks.Layer
   nengo_extras.deepnetworks.NodeLayer
   nengo_extras.deepnetworks.NeuronLayer
   nengo_extras.deepnetworks.DataLayer
   nengo_extras.deepnetworks.SoftmaxLayer
   nengo_extras.deepnetworks.DropoutLayer
   nengo_extras.deepnetworks.FullLayer
   nengo_extras.deepnetworks.ProcessLayer
   nengo_extras.deepnetworks.LocalLayer
   nengo_extras.deepnetworks.ConvLayer
   nengo_extras.deepnetworks.PoolLayer
   nengo_extras.cuda_convnet.CudaConvnetNetwork

.. autoclass:: nengo_extras.deepnetworks.DeepNetwork

.. autoclass:: nengo_extras.deepnetworks.SequentialNetwork

.. autoclass:: nengo_extras.keras.SequentialNetwork

.. autoclass:: nengo_extras.deepnetworks.Layer

.. autoclass:: nengo_extras.deepnetworks.NodeLayer

.. autoclass:: nengo_extras.deepnetworks.NeuronLayer

.. autoclass:: nengo_extras.deepnetworks.DataLayer

.. autoclass:: nengo_extras.deepnetworks.SoftmaxLayer

.. autoclass:: nengo_extras.deepnetworks.DropoutLayer

.. autoclass:: nengo_extras.deepnetworks.FullLayer

.. autoclass:: nengo_extras.deepnetworks.ProcessLayer

.. autoclass:: nengo_extras.deepnetworks.LocalLayer

.. autoclass:: nengo_extras.deepnetworks.ConvLayer

.. autoclass:: nengo_extras.deepnetworks.PoolLayer

.. autoclass:: nengo_extras.cuda_convnet.CudaConvnetNetwork

Processes
=========

.. autoclass:: nengo_extras.convnet.Conv2d
   :no-members:

.. autoclass:: nengo_extras.convnet.Pool2d
   :no-members:
