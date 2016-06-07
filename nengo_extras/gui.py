from __future__ import absolute_import

import nengo

from .convnet import ShapeParam


def preprocess_display(x, transpose=(1, 2, 0), scale=255., offset=0.):
    """Basic preprocessing that reshapes, transposes, and scales an image"""
    x = (x + offset) * scale
    if transpose is not None:
        x = x.transpose(transpose)  # color channel last
    if x.shape[-1] == 1:
        x = x[..., 0]
    return x.clip(0, 255).astype('uint8')


def image_html_function(image_shape, preprocess=preprocess_display,
                         **preprocess_args):
    """Make a function to turn an image into HTML to display as an SVG.

    Parameters
    ----------
    image_shape : array_like (3,)
        The shape of the image: (channels, height, width)
    preprocess : callable
        Callable that takes an image and preprocesses it to be displayed.
    preprocess_args : dict
        Optional dictionary of keyword arguments for ``preprocess``.

    Returns
    -------
    html_function : callable (t, x)
        A function that takes time and a flattened image, and returns a string
        that defines an SVG object in HTML to display the image.
    """
    import base64
    import PIL.Image
    import cStringIO

    assert len(image_shape) == 3

    def html_function(x):
        x = x.reshape(image_shape)
        y = preprocess(x, **preprocess_args)
        png = PIL.Image.fromarray(y)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    return html_function


def image_display_function(image_shape, preprocess=preprocess_display,
                           **preprocess_args):
    """Make a function to display images in Nengo GUI

    Examples
    --------
    >>> u = nengo.Node(nengo.processes.PresentInput(images, 0.1))
    >>> display_f = nengo_extras.gui.image_display_function(image_shape)
    >>> display_node = nengo.Node(display_f, size_in=u.size_out)
    >>> nengo.Connection(u, display_node, synapse=None)

    Requirements
    ------------
    pillow (provides PIL, `pip install pillow`)
    """
    html_function = image_html_function(
        image_shape, preprocess=preprocess, **preprocess_args)

    def display_func(t, x):
        display_func._nengo_html_ = html_function(x)

    return display_func


class PresentImages(nengo.processes.PresentInput):
    """PresentInput process whose inputs are displayed as images in nengo_gui.
    """

    image_shape = ShapeParam('image_shape', length=3, low=1)

    def __init__(self, images, presentation_time, **kwargs):
        self.image_shape = images.shape[1:]
        super(PresentImages, self).__init__(
            images, presentation_time, **kwargs)
        self.configure_display()

    def configure_display(self, preprocess=preprocess_display,
                          **preprocess_args):
        """Configure display parameters for images

        Parameters
        ----------
        preprocess : callable
            Callable that takes an image and preprocesses it to be displayed.
        preprocess_args : dict
            Optional dictionary of keyword arguments for ``preprocess``.
        """
        html_function = image_html_function(
            self.image_shape, preprocess=preprocess, **preprocess_args)

        def _nengo_html_(t, x):
            return html_function(x)

        self._nengo_html_ = _nengo_html_
