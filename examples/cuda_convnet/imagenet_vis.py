import os

from nengo_extras.data import load_ilsvrc2012
from nengo_extras.cuda_convnet import CudaConvnetNetwork, load_model_pickle
from nengo_extras.gui import image_function
from nengo_extras.deepnetworks import ConvLayer, PoolLayer, NeuronLayer
from nengo_extras.deepview import Viewer

os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'


def preprocess(x, scale=255, offset=0):
    if x.shape[0] != 3:
        x = x.mean(axis=0, keepdims=True)
    x = (x + offset) * scale
    x = x.transpose((1, 2, 0))  # color channel last
    if x.shape[-1] == 1:
        x = x[..., 0]
    return x.clip(0, 255).astype('uint8')


def get_filters_image_fn(filters):
    filters = layer.process.filters
    fmean, fwidth = filters.mean(), 2*filters.std()
    filter_image_fn = image_function(
        filters.shape[1:], preprocess=preprocess,
        offset=fwidth-fmean, scale=128./fwidth)
    return filter_image_fn


def get_act_image_fn(acts):
    act_shape = (1,) + acts.shape[2:]
    amean, awidth = acts.mean(), 2*acts.std()
    act_image_fn = image_function(
        act_shape, preprocess=preprocess,
        offset=awidth-amean, scale=128./awidth)
    return act_image_fn


# --- load data
# retrieve from https://figshare.com/s/cdde71007405eb11a88f
filename = 'ilsvrc-2012-batches-test3.tar.gz'
X_test, Y_test, data_mean, label_names = load_ilsvrc2012(filename, n_files=1)

X_test = X_test.astype('float32')

# crop data
X_test = X_test[:, :, 16:-16, 16:-16]
data_mean = data_mean[:, 16:-16, 16:-16]
image_shape = X_test.shape[1:]

# subtract mean
X_test -= data_mean

# retrieve from https://figshare.com/s/f343c68df647e675af28
model_filename = 'ilsvrc2012-lif-48.pkl'
cc_model = load_model_pickle(model_filename)

# --- forward pass
n = 100
images = X_test[:n]

ccnet = CudaConvnetNetwork(cc_model)

y = images.reshape((images.shape[0], -1))
outputs = []
for layer in ccnet.layers:
    y = layer.theano_compute(y)
    outputs.append(y)

# --- create viewer
input_image_fn = image_function(image_shape, offset=data_mean, scale=1)
viewer = Viewer(images, input_image_fn)
viewer.wm_title(model_filename)

k = 1
layers = ccnet.layers
while k < len(outputs):
    layer = layers[k]
    layer1 = layers[k+1] if k+1 < len(layers) else None
    layer2 = layers[k+2] if k+2 < len(layers) else None

    if isinstance(layer, ConvLayer) and isinstance(layer1, NeuronLayer) and (
            isinstance(layer2, PoolLayer)):
        filters = layer.process.filters
        out = outputs[k].reshape((-1,) + layer.shape_out)
        out1 = outputs[k+1].reshape((-1,) + layer.shape_out)
        out2 = outputs[k+2].reshape((-1,) + layer2.shape_out)
        filter_fn = get_filters_image_fn(filters)
        fn = get_act_image_fn(out)
        fn1 = get_act_image_fn(out1)
        fn2 = get_act_image_fn(out2)
        title = "%s/%s/%s" % (layer.label, layer1.label, layer2.label)
        viewer.add_column(
            ('filters', 'acts', 'acts', 'acts'),
            (filters, out, out1, out2),
            (filter_fn, fn, fn1, fn2),
            title=title)
        k += 2
    elif isinstance(layer, ConvLayer) and isinstance(layer1, NeuronLayer):
        filters = layer.process.filters
        out = outputs[k].reshape((-1,) + layer.shape_out)
        out1 = outputs[k+1].reshape((-1,) + layer.shape_out)
        filter_fn = get_filters_image_fn(filters)
        fn = get_act_image_fn(out)
        fn1 = get_act_image_fn(out1)
        title = "%s/%s" % (layer.label, layer1.label)
        viewer.add_column(
            ('filters', 'acts', 'acts'),
            (filters, out, out1),
            (filter_fn, fn, fn1),
            title=title)
        k += 1
    elif isinstance(layer, ConvLayer):
        filters = layer.process.filters
        out = outputs[k].reshape((-1,) + layer.shape_out)
        filter_fn = get_filters_image_fn(filters)
        fn = get_act_image_fn(out)
        viewer.add_filters_acts(filters, filter_fn, out, fn, title=layer.label)
    elif isinstance(layer, PoolLayer):
        out = outputs[k].reshape((-1,) + layer.shape_out)
        fn = get_act_image_fn(out)
        viewer.add_acts(out, fn, title=layer.label)
    else:
        print("Skipping %r" % layer.label)

    k += 1


viewer.set_index(0)
viewer.mainloop()
