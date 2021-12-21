from __future__ import absolute_import

import base64
from io import BytesIO

import PIL.Image


def preprocess_display(x, transpose=(1, 2, 0), scale=255.0, offset=0.0):
    """Basic preprocessing that reshapes, transposes, and scales an image"""
    x = (x + offset) * scale
    if transpose is not None:
        x = x.transpose(transpose)  # color channel last
    if x.shape[-1] == 1:
        x = x[..., 0]
    return x.clip(0, 255).astype("uint8")


def image_function(image_shape, preprocess=preprocess_display, **preprocess_args):
    """Make a function to turn an array into a PIL Image.

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
    to_pil : callable (x)
        A function that takes a (flattened) image array, returning a PIL Image.
    """
    assert len(image_shape) == 3

    def to_pil(x):
        x = x.reshape(image_shape)
        y = preprocess(x, **preprocess_args)
        image = PIL.Image.fromarray(y)
        return image

    return to_pil


def image_string_function(
    image_shape, format="PNG", preprocess=preprocess_display, **preprocess_args
):
    """Make a function to turn an array into an image string.

    Parameters
    ----------
    image_shape : array_like (3,)
        The shape of the image: (channels, height, width)
    format : string
        A format string for the ``PIL.Image.save`` function.
    preprocess : callable
        Callable that takes an image and preprocesses it to be displayed.
    preprocess_args : dict
        Optional dictionary of keyword arguments for ``preprocess``.

    Returns
    -------
    string_function : callable (x)
        A function that takes a (flattened) image array, and returns a
        base64 string representation of the image of the requested format.

    See also
    --------
    image_function
    """
    to_pil = image_function(image_shape, preprocess=preprocess, **preprocess_args)

    def string_function(x):
        image = to_pil(x)
        buffer = BytesIO()
        image.save(buffer, format=format)
        image_string = base64.b64encode(buffer.getvalue()).decode()
        return image_string

    return string_function


def image_html_function(image_shape, preprocess=preprocess_display, **preprocess_args):
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
    html_function : callable (x)
        A function that takes a (flattened) image array, and returns
        a string that defines an SVG object in HTML to display the image.

    See also
    --------
    image_function
    """
    string_function = image_string_function(
        image_shape, preprocess=preprocess, **preprocess_args
    )

    def html_function(x):
        image_string = string_function(x)
        return """
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>""" % (
            "".join(image_string)
        )

    return html_function


def image_display_function(
    image_shape, preprocess=preprocess_display, **preprocess_args
):
    """Make a function to display images in Nengo GUI.

    Examples
    --------

    Displaying images from a ``PresentInput`` process in NengoGUI.

    .. testcode::

       from nengo_extras.gui import image_display_function

       with nengo.Network() as net:
           u = nengo.Node(nengo.processes.PresentInput([[0], [0.5], [1.0]], 0.1))
           display_f = image_display_function((1, 1, 1))
           display_node = nengo.Node(display_f, size_in=u.size_out)
           nengo.Connection(u, display_node, synapse=None)
    """
    html_function = image_html_function(
        image_shape, preprocess=preprocess, **preprocess_args
    )

    def display_func(t, x):
        display_func._nengo_html_ = html_function(x)

    return display_func
