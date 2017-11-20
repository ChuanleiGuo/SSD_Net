import copy
from math import ceil, floor
import numpy as np
import mxnet as mx

from networks.symbol_factory import get_symbol_train

rolling_time = 4
rolling_rate = 0.075
gpus = "0,1,2,3"
gpulist = gpus.split(",")
num_gpus = len(gpulist)
batch_size = 4
num_classes = 2

resize_height, resize_width = 2560, 768
min_dim = min(resize_width, resize_height)

# in percent %
min_ratio = 15
max_ratio = 85
# TODO: change the implementation of using hard code
step = int(floor((max_ratio - min_ratio) / (5 - 2)))
min_sizes = []
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 6.7 / 100.] + min_sizes
max_sizes = [[]] + max_sizes
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3]]

def create_rolling_struct(net, from_layers_basename=[], num_outputs=[], odd=[],
        rolling_rate=0.25, roll_idx=1, conv2=False, normalize=True):

    roll_layers = []
    factor = 2
    from_layers = copy.copy(from_layers_basename)
    assert len(from_layers) == len(num_outputs)

    if roll_idx == 1:
        if normalize:
            from_layers[0] = "%s_norm" % from_layers[0]
    else:
        for i in range(len(from_layers)):
            from_layers[i] = "%s_%d" % (from_layers[i], roll_idx)
    for i in range(len(from_layers)):
        internals = net.get_internals()
        f_layers = []
        num_out = int(num_outputs[i] * rolling_rate)

        if i > 0:
            f_layer = from_layers[i - 1]
            o_layer_name = "%s_r%d" % (from_layers_basename[i], roll_idx)
            bias = mx.sym.Variable(
                name=o_layer_name+"_bias",
                init=mx.init.Constant(0.0),
                attr={
                    '__lr_mult__': '2.0'
                })
            o_layer = mx.sym.Convolution(data=internals[f_layer],
                num_filter=num_out, kernel=(1, 1), stride=(1, 1),
                name=o_layer_name, bias=bias)
            o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)
            o_layer = mx.sym.Pooling(data=o_layer, pool_type="max", kernel=(2, 2),
                stride=(2, 2), name="pool_" + o_layer_name)

            f_layers.append(o_layer)

        f_layers.append(internals[from_layers[i]])

        if i < len(from_layers) - 1:
            f_layer = from_layers[i + 1]
            o_layer_name = "%s_l%d" % (from_layers_basename[i + 1], roll_idx)
            bias = mx.sym.Variable(
                name=o_layer_name+"_bias",
                init=mx.init.Constant(0.0),
                attr={
                    '__lr_mult__': '2.0'
                })
            o_layer = mx.sym.Convolution(data=internals[f_layer],
                num_filter=num_out, kernel=(1, 1), stride=(1, 1),
                name=o_layer_name, bias=bias)
            o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)
            f_layer = o_layer_name

            if odd[i]:
                o_layer_name = "%s_deconv" % f_layer
                o_layer = mx.sym.Deconvolution(data=o_layer, num_filter=num_out,
                    num_group=num_out, kernel=int(2 * factor - factor % 2),
                    pad=int(np.ceil((factor - 1) / 2.)), stride=int(factor),
                    name=o_layer_name, no_bias=True)
                temp_layer = f_layer
                f_layer = o_layer_name
                o_layer_name = "%s_deconv" % temp_layer
                if not conv2:
                    o_layer = mx.sym.Pooling(data=o_layer, pool_type="avg",
                        kernel=2, stride=1, name=o_layer_name)
                else:
                    bias = mx.sym.Variable(
                        name=o_layer_name+"_bias",
                        init=mx.init.Constant(0.0),
                        attr={
                            '__lr_mult__': '2.0'
                        })
                    o_layer = mx.sym.Convolution(o_layer, num_filter=num_out,
                        name=o_layer_name, bias=bias)
                    o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)
            else:
                o_layer_name = "%s_deconv" % f_layer
                o_layer = mx.sym.Deconvolution(data=o_layer, num_filter=num_out,
                    num_group=num_out, kernel=int(2 * factor - factor % 2),
                    pad=int(ceil((factor - 1) / 2.)), stride=int(factor),
                    name=o_layer_name, no_bias=False)
            f_layer.append(o_layer)

        o_layer_name = "%s_concat_%s" % (from_layers_basename[i], roll_idx)
        o_layer = mx.sym.concat(*f_layers, dim=1, name=o_layer_name)

        o_layer_name = "%s_%d" % (from_layers_basename[i], roll_idx + 1)
        bias = mx.sym.Variable(
            name=o_layer_name+"_bias",
            init=mx.init.Constant(0.0),
            attr={
                '__lr_mult__': '2.0'
            })
        o_layer = mx.sym.Convolution(data=o_layer, num_filter=num_outputs[i],
            kernel=1, stride=1, name=o_layer_name, bias=bias)
        o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)

        roll_layers.append(o_layer)

    return roll_layers
