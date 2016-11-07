def preprocess_display(x, image_shape,
                       transpose=(1, 2, 0), scale=255., offset=0.):
    """Basic preprocessing that reshapes, transposes, and scales an image"""
    y = x.reshape(image_shape)
    y = (y + offset) * scale
    if transpose is not None:
        y = y.transpose(transpose)  # color channel last
    if y.shape[-1] == 1:
        y = y[..., 0]
    return y.clip(0, 255).astype('uint8')


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
    import base64
    import PIL.Image
    import cStringIO

    assert len(image_shape) == 3

    def display_func(t, x):
        y = preprocess(x, image_shape, **preprocess_args)
        png = PIL.Image.fromarray(y)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())

        display_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    return display_func
