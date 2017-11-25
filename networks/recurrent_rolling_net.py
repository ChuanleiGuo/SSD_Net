from __future__ import absolute_import
import copy
from math import ceil, floor
import numpy as np
import mxnet as mx
from .common import multi_layer_feature, multibox_layer


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


rolling_rate = 0.075

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

normalizations = [20, -1, -1, -1, -1]
normalizations2 = [-1, -1, -1, -1, -1]
num_outputs=[256,256,256,256,256]
odd=[0,0,0,0,0]
rolling_rate = 0.075

def create_rolling_struct(from_layers=[], num_outputs=[], odd=[],
        rolling_rate=0.25, roll_idx=1, conv2=False, normalize=True):

    """ Build rolling structure between `from_layers`
    ## Prameter
    from_layers: mx.symbol
        layers used to contruct rolling structure
    num_outputs: ints
        filters in different layers, same length of `from_layers`
    odd:

    rolling_rate: float
    roll_idx: rolling_idx to start
    conv2: whether using conv while rolling
    normalize: whether to normalize layers
    """

    roll_layers = []
    factor = 2
    from_layer_names = [l.name for l in from_layers]
    assert len(from_layer_names) == len(num_outputs)

    if roll_idx == 1:
        if normalize:
            from_layer_names[0] = "%s_norm" % from_layers[0].name
    else:
        for i in range(len(from_layer_names)):
            from_layer_names[i] = "%s_%d" % (from_layers[i].name, roll_idx)
    for i in range(len(from_layers)):
        f_layers = []
        num_out = int(num_outputs[i] * rolling_rate)

        if i > 0:
            f_layer = from_layers[i - 1]
            o_layer_name = "%s_r%d" % (from_layer_names[i], roll_idx)
            bias = mx.sym.Variable(
                name=o_layer_name+"_bias",
                init=mx.init.Constant(0.0),
                attr={
                    '__lr_mult__': '2.0'
                })
            o_layer = mx.sym.Convolution(data=f_layer,
                num_filter=num_out, kernel=(1, 1), stride=(1, 1),
                name=o_layer_name, bias=bias)
            o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)
            o_layer = mx.sym.Pooling(data=o_layer, pool_type="max", kernel=(2, 2),
                stride=(2, 2), name="pool_" + o_layer_name)

            f_layers.append(o_layer)

        f_layers.append(from_layers[i])

        if i < len(from_layers) - 1:
            f_layer = from_layers[i + 1]
            o_layer_name = "%s_l%d" % (from_layer_names[i + 1], roll_idx)
            bias = mx.sym.Variable(
                name=o_layer_name+"_bias",
                init=mx.init.Constant(0.0),
                attr={
                    '__lr_mult__': '2.0'
                })
            o_layer = mx.sym.Convolution(data=f_layer,
                num_filter=num_out, kernel=(1, 1), stride=(1, 1),
                name=o_layer_name, bias=bias)
            o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)

            f_layer = o_layer

            if odd[i]:
                o_layer_name = "%s_deconv" % f_layer.name
                o_layer = mx.sym.Deconvolution(data=o_layer, num_filter=num_out,
                    num_group=num_out, kernel=int(2 * factor - factor % 2),
                    pad=int(np.ceil((factor - 1) / 2.)), stride=int(factor),
                    name=o_layer_name, no_bias=True)
                temp_layer = f_layer
                f_layer = o_layer
                o_layer_name = "%s_deconv" % temp_layer.name
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
                o_layer_name = "%s_deconv" % f_layer.name
                o_layer = mx.sym.Deconvolution(data=o_layer, num_filter=num_out,
                    num_group=num_out, kernel=int(2 * factor - factor % 2),
                    pad=int(ceil((factor - 1) / 2.)), stride=int(factor),
                    name=o_layer_name, no_bias=False)
            f_layers.append(o_layer)

        o_layer_name = "%s_concat_%s" % (from_layer_names[i], roll_idx)
        o_layer = mx.sym.concat(*f_layers, dim=1, name=o_layer_name)

        o_layer_name = "%s_%d" % (from_layer_names[i], roll_idx + 1)
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

def add_multibox_and_loss_for_extra(extra_layers, label, num_classes, num_filters,
        sizes, ratios, normalizations=-1, steps=[], nms_thresh=0.5,
        force_suppress=False, nms_topk=400, rolling_idx=0):

    loc_preds, cls_preds, anchor_boxes = multibox_layer(extra_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target_%d" % rolling_idx)
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob_%d" % rolling_idx)
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss__%d" % rolling_idx, \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss_%d" % rolling_idx)

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(
        data=cls_target, grad_scale=0, name="cls_label_%d" % rolling_idx)
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection_%d" % rolling_idx, nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out_%d" % rolling_idx)

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])

    return out

def add_multibox_for_extra(extra_layers, num_classes, num_filters,
        sizes, ratios, normalizations=-1, steps=[], nms_thresh=0.5,
        force_suppress=False, nms_topk=400, rolling_idx=0):

    loc_preds, cls_preds, anchor_boxes = multibox_layer(extra_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob_%d' % rolling_idx)
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection_%d" % rolling_idx, nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    return out


def get_symbol_rolling_train(
                            rolling_time,
                            network,
                            num_classes,
                            from_layers,
                            num_filters,
                            strides,
                            pads,
                            sizes,
                            ratios,
                            normalizations=-1,
                            steps=[],
                            min_filter=128,
                            nms_thresh=0.5,
                            force_suppress=False,
                            nms_topk=400,
                            **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(
        body, from_layers, num_filters, strides, pads, min_filter=min_filter)

    # group output
    out = add_multibox_and_loss_for_extra(layers, label=label, num_classes=num_classes,
        num_filters=num_filters, sizes=sizes, ratios=ratios, normalizations=normalizations,
        steps=steps, nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk,
        rolling_idx=0)

    outputs = [out]

    # Rolling Layers
    for roll_idx in range(1, rolling_time + 1):
        roll_layers = create_rolling_struct(layers, num_outputs=[256] * 7, odd=[0] * 7,
            rolling_rate=rolling_rate, roll_idx=roll_idx, conv2=False, normalize=True)
        out = add_multibox_and_loss_for_extra(roll_layers, label=label, num_classes=num_classes,
            num_filters=num_filters, sizes=sizes, ratios=ratios, normalizations=normalizations,
            steps=steps, nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk,
            rolling_idx=roll_idx)

        outputs.append(out)

    return mx.symbol.Group(outputs)

def get_symbol_rolling_test(
                rolling_time,
                network,
                num_classes,
                from_layers,
                num_filters,
                sizes,
                ratios,
                strides,
                pads,
                normalizations=-1,
                steps=[],
                min_filter=128,
                nms_thresh=0.5,
                force_suppress=False,
                nms_topk=400,
                **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(
        body, from_layers, num_filters, strides, pads, min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    outputs = [out]

    for roll_idx in range(1, rolling_time + 1):
        roll_layers = create_rolling_struct(layers, num_outputs=num_outputs, odd=odd,
            rolling_rate=rolling_rate, roll_idx=roll_idx, conv2=False, normalize=True)
        out = add_multibox_for_extra(roll_layers, num_classes=num_classes,
            num_filters=num_filters, sizes=sizes, ratios=ratios, normalizations=normalizations,
            steps=steps, nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk,
            rolling_idx=roll_idx)

        outputs.append(out)

    return mx.sym.Group(outputs)
