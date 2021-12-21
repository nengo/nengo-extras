*************
Vision models
*************

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/mnist_single_layer

.. autosummary::

   nengo_extras.vision.Gabor
   nengo_extras.vision.Mask
   nengo_extras.vision.ciw_encoders
   nengo_extras.vision.cd_encoders_biases
   nengo_extras.vision.percentile_biases

.. autoclass:: nengo_extras.vision.Gabor

.. autoclass:: nengo_extras.vision.Mask

.. autofunction:: nengo_extras.vision.ciw_encoders

.. autofunction:: nengo_extras.vision.cd_encoders_biases

.. autofunction:: nengo_extras.vision.percentile_biases

.. autoclass:: nengo_extras.convnet.PresentJitteredImages
  :no-members:

Camera input
============

To use these classes, you will have to install
GStreamer and some Python dependencies:

.. code-block:: bash

   sudo apt install python-gst-1.0
   pip install vext vext.gi

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/cuda_convnet/webcam_spiking_cnn

.. autosummary::

   nengo_extras.camera.CameraPipeline
   nengo_extras.camera.CameraData
   nengo_extras.camera.Camera

.. autoclass:: nengo_extras.camera.CameraPipeline

.. autoclass:: nengo_extras.camera.CameraData

.. autoclass:: nengo_extras.camera.Camera
   :no-members:
