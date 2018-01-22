# -*- coding: utf-8 -*-
from __future__ import absolute_import
import copy
from math import ceil, floor
import numpy as np
import mxnet as mx
from networks.common import conv_act_layer, multibox_layer, branched_multibox_layer


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


resize_height, resize_width = 2560, 768
min_dim = min(resize_width, resize_height)

# in percent %
min_ratio = 15
max_ratio = 85
rolling_rate = 0.075


def _get_sym_output_shape(sym, input_shape):
    _, out_shapes, _ = sym.infer_shape(data=input_shape)
    return out_shapes[0][2:]


def _get_shared_weights(num_layers, strides):
    forward_weights = []
    for i in range(num_layers - 1):
        weight1x1 = mx.sym.Variable(
            name="forward_1x1_%d_weight" % i,
            lr_mult=1,
            wd_mult=1,
            init=mx.init.Xavier())
        bias1x1 = mx.sym.Variable(
            name="forward_1x1_%d_bias" % i,
            lr_mult=2,
            wd_mult=0,
            init=mx.init.Constant(0))
        weight3x3 = mx.sym.Variable(
            name="forward_3x3_%d_weight" % i,
            lr_mult=1,
            wd_mult=1,
            init=mx.init.Xavier())
        bias3x3 = mx.sym.Variable(
            name="forward_3x3_%d_bias" % i,
            lr_mult=2,
            wd_mult=0,
            init=mx.init.Constant(0))
        forward_weights.append((weight1x1, bias1x1, weight3x3, bias3x3))

    backward_weights = []
    deconv_weights = []
    for i in range(num_layers - 1):
        weight = mx.sym.Variable(
            name="backward_%d_weight" % i,
            lr_mult=1,
            wd_mult=1,
            init=mx.init.Xavier())
        bias = mx.sym.Variable(
            name="backward_%d_bias" % i,
            lr_mult=2,
            wd_mult=0,
            init=mx.init.Constant(0))
        backward_weights.append((weight, bias))

        dweight = mx.sym.Variable(
            name="deconv_%d_weight" % i,
            lr_mult=1,
            wd_mult=1,
            init=mx.init.Xavier())
        dbias = mx.sym.Variable(
            name="deconv_%d_bias" % i,
            lr_mult=2,
            wd_mult=0,
            init=mx.init.Constant(0))
        deconv_weights.append((dweight, dbias))

    concat_weights = []
    for i in range(num_layers):
        weight = mx.sym.Variable(
            name="concat_%d_weight" % i,
            lr_mult=1,
            wd_mult=1,
            init=mx.init.Xavier())
        bias = mx.sym.Variable(
            name="concat_%d_bias" % i,
            lr_mult=2,
            wd_mult=0,
            init=mx.init.Constant(0))
        concat_weights.append((weight, bias))

    return (forward_weights, backward_weights, deconv_weights, concat_weights)


def multi_layer_feature(body,
                        from_layers,
                        num_filters,
                        strides,
                        pads,
                        min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
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
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    internals = body.get_internals()
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            #layers.append(layer)
            # num1x1 = max(min_filter, num_filter // 2)
            # conv_1x1 = conv_act_layer(
            #     layer,
            #     "multi_feat_%d_conv_1x1" % (k),
            #     num1x1,
            #     kernel=(1, 1),
            #     pad=(0, 0),
            #     stride=(1, 1),
            #     act_type="relu")
            conv_3x3 = conv_act_layer(
                layer,
                "multi_feat_%d_conv_3x3" % (k),
                num_filter,
                kernel=(3, 3),
                pad=(1, 1),
                stride=(1, 1),
                act_type="relu")
            layers.append(conv_3x3)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(
                layer,
                'multi_feat_%d_conv_1x1' % (k),
                num_1x1,
                kernel=(1, 1),
                pad=(0, 0),
                stride=(1, 1),
                act_type='relu')
            conv_3x3 = conv_act_layer(
                conv_1x1,
                'multi_feat_%d_conv_3x3' % (k),
                num_filter,
                kernel=(3, 3),
                pad=(p, p),
                stride=(s, s),
                act_type='relu')
            layers.append(conv_3x3)
    return layers


def create_rolling_struct(from_layers,
                          data_shape,
                          num_filters,
                          strides,
                          pads,
                          rolling_rate,
                          roll_idx,
                          conv2=False,
                          normalize=True,
                          shared_weights=None):
    # strides 为 -1 时，实现方法，根据两层之间的尺寸比值
    rolling_layers = []
    from_layer_names = [l.name for l in from_layers]
    assert len(from_layer_names) == len(num_filters)

    if roll_idx == 1:
        if normalize:
            from_layer_names[0] = "%s_norm" % (from_layer_names[0])
    else:
        for i in range(len(from_layer_names)):
            from_layer_names[i] = "%s_%d" % (from_layer_names[i], roll_idx)

    forward_weights, backward_weights, deconv_weights, concat_weights = shared_weights

    for i in range(len(from_layers)):
        f_layers = []
        num_filter = int(num_filters[i] * rolling_rate)

        if i > 0:
            f_layer = from_layers[i - 1]
            o_layer_name = "%s_r%d" % (from_layer_names[i - 1], roll_idx)

            f_weight1x1, f_bias1x1, f_weight3x3, f_bias3x3 = forward_weights[
                i - 1]

            if strides[i] == -1:

                o_layer = mx.sym.Convolution(data=f_layer, weight=f_weight1x1, bias=f_bias1x1, \
                    num_filter=num_filter, stride=(1, 1), pad=(0, 0), kernel=(1, 1), \
                    name=o_layer_name)
                o_layer = mx.sym.relu(
                    data=o_layer, name="relu1_" + o_layer_name)
                o_layer = mx.sym.Convolution(data=o_layer, weight=f_weight3x3, bias=f_bias3x3, \
                    num_filter=num_filter, stride=(1, 1), pad=(1, 1), kernel=(3, 3), \
                    name="conv3x3_" + o_layer_name)
                o_layer = mx.sym.relu(
                    data=o_layer, name="relu3_" + o_layer_name)
                o_layer = mx.sym.Pooling(
                    data=o_layer,
                    pool_type="max",
                    kernel=(2, 2),
                    stride=(2, 2),
                    name="pool_" + o_layer_name)
            else:

                o_layer = mx.sym.Convolution(data=f_layer, weight=f_weight1x1, bias=f_bias1x1, \
                    num_filter=num_filter, stride=(1, 1), pad=(0, 0), kernel=(1, 1), \
                    name=o_layer_name)
                o_layer = mx.sym.relu(
                    data=o_layer, name="relu1_" + o_layer_name)
                s, p = strides[i], pads[i]
                o_layer = mx.sym.Convolution(data=o_layer, weight=f_weight3x3, bias=f_bias3x3, \
                    num_filter=num_filter, stride=(s, s), pad=(p, p), kernel=(3, 3), \
                    name="conv3x3_" + o_layer_name)
                o_layer = mx.sym.relu(
                    data=o_layer, name="relu3_" + o_layer_name)

            f_layers.append(o_layer)

        f_layers.append(from_layers[i])

        if i < len(from_layers) - 1:
            f_layer = from_layers[i + 1]
            o_layer_name = "%s_l%d" % (from_layer_names[i + 1], roll_idx)

            b_weight, b_bias = backward_weights[i]

            o_layer = mx.sym.Convolution(data=f_layer, weight=b_weight, bias=b_bias, \
                num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), \
                name=o_layer_name)
            o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)

            next_layer_output_size = _get_sym_output_shape(
                f_layer, (20, 3, data_shape[1], data_shape[2]))
            layer_output_size = _get_sym_output_shape(
                from_layers[i], (20, 3, data_shape[1], data_shape[2]))


            if data_shape[1] == data_shape[2]:

                isTwice = (layer_output_size[0] / next_layer_output_size[0] == 2) and \
                    (layer_output_size[1] / next_layer_output_size[1] == 2)
                if strides[i + 1] == -1 or isTwice:
                    factor = 2
                    p = int(ceil((factor - 1) / 2.))
                    k = int(2 * factor - factor % 2)
                    s = int(factor)
                else:
                    s, p = strides[i + 1], pads[i + 1]
                    k = 3

                s = (s, s)
                p = (p, p)
                k = (k, k)
            else:
                factor = 2
                h_twice = layer_output_size[0] / next_layer_output_size[0] == 2
                w_twice = layer_output_size[1] / next_layer_output_size[1] == 2
                h_p = int(ceil((factor - 1) / 2.)) if (strides[i + 1] == -1 or h_twice) \
                    else pads[i + 1]
                w_p = int(ceil((factor - 1) / 2.)) if (strides[i + 1] == -1 or w_twice) \
                    else pads[i + 1]
                h_s = int(factor) if (strides[i + 1] == -1 or h_twice) else strides[i + 1]
                w_s = int(factor) if (strides[i + 1] == -1 or w_twice) else strides[i + 1]
                h_k = int(2 * factor - factor % 2) if (strides[i + 1] == -1 or h_twice) else 3
                w_k = int(2 * factor - factor % 2) if (strides[i + 1] == -1 or w_twice) else 3

                s = (h_s, w_s)
                p = (h_p, w_p)
                k = (h_k, w_k)

            d_weight, _ = deconv_weights[i]

            o_layer = mx.sym.Deconvolution(data=o_layer, weight=d_weight, no_bias=True, \
                num_filter=num_filter, kernel=k, stride=s, pad=p, \
                name="deconv_"+o_layer_name)
            # o_layer = mx.sym.relu(data=o_layer, name="relu_deconv_" + o_layer_name)

            f_layers.append(o_layer)

        o_layer_name = "%s_concat_%s" % (from_layer_names[i], roll_idx)
        o_layer = mx.sym.concat(*f_layers, dim=1)
        o_layer_name = "%s_%d" % (from_layer_names[i], roll_idx + 1)

        c_weight, c_bias = concat_weights[i]

        o_layer = mx.sym.Convolution(data=o_layer, weight=c_weight, bias=c_bias, \
            num_filter=num_filters[i], kernel=(1, 1), stride=(1, 1), pad=(0, 0), \
            name=o_layer_name)
        o_layer = mx.sym.relu(data=o_layer, name="relu_" + o_layer_name)

        rolling_layers.append(o_layer)

    return rolling_layers


def add_multibox_and_loss_for_extra(extra_layers,
                                    label,
                                    num_classes,
                                    num_filters,
                                    sizes,
                                    ratios,
                                    normalizations=-1,
                                    steps=[],
                                    nms_thresh=0.5,
                                    force_suppress=False,
                                    nms_topk=400,
                                    rolling_idx=0,
                                    mbox_shared_weights=None):
    if len(sizes) == len(extra_layers):
        loc_preds, cls_preds, anchor_boxes = multibox_layer(extra_layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    elif len(sizes) > len(extra_layers):
        loc_preds, cls_preds, anchor_boxes = branched_multibox_layer(extra_layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps, \
            branch_num=4, shared_weights=mbox_shared_weights)
    else:
        raise ValueError("wrong number of sizes")

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target_%d" % rolling_idx if rolling_idx else "multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob_%d" % rolling_idx if rolling_idx else "cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(
        name="loc_loss__%d" % rolling_idx if rolling_idx else "loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target),
        scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss_%d" % rolling_idx if rolling_idx else "loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(
        data=cls_target,
        grad_scale=0,
        name="cls_label_%d" % rolling_idx if rolling_idx else "cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(
        *[cls_prob, loc_preds, anchor_boxes],
        name="detection_%d" % rolling_idx if rolling_idx else "detection",
        nms_threshold=nms_thresh,
        force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2),
        nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(
        data=det,
        grad_scale=0,
        name="det_out_%d" % rolling_idx if rolling_idx else "det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])

    return out


def add_multibox_for_extra(extra_layers,
                           num_classes,
                           num_filters,
                           sizes,
                           ratios,
                           normalizations=-1,
                           steps=[],
                           nms_thresh=0.5,
                           force_suppress=False,
                           nms_topk=400,
                           rolling_idx=0,
                           mbox_shared_weights=None):
    if len(sizes) > len(extra_layers):
        loc_preds, cls_preds, anchor_boxes = branched_multibox_layer(extra_layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps, branch_num=4, \
            shared_weights=mbox_shared_weights)
    elif len(sizes) == len(extra_layers):
        loc_preds, cls_preds, anchor_boxes = multibox_layer(extra_layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    else:
        raise ValueError("Wrong number of sizes")

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob_%d' % rolling_idx if rolling_idx else "cls_prob")
    out = mx.contrib.symbol.MultiBoxDetection(
        *[cls_prob, loc_preds, anchor_boxes],
        name="detection_%d" % rolling_idx if rolling_idx else "detection",
        nms_threshold=nms_thresh,
        force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2),
        nms_topk=nms_topk)

    return out


def _get_multibox_shared_weights(num_layers, branch_num):
    shared_weights = []
    for layer_idx in range(num_layers):
        layer_mbox_weights = []
        if layer_idx == num_layers - 1:
            branch_num = 1
        for branch_idx in range(branch_num):
            loc_weight = mx.sym.Variable(
                name="layer_{}_shared_mbox_loc_{}_weight".format(layer_idx, branch_idx),
                lr_mult=1,
                wd_mult=1,
                init=mx.init.Xavier()
            )
            loc_bias = mx.sym.Variable(
                name="layer_{}_shared_mbox_loc_{}_bias".format(layer_idx, branch_idx),
                lr_mult=2,
                wd_mult=0,
                init=mx.init.Constant(0)
            )

            conf_weight = mx.sym.Variable(
                name="layer_{}_shared_mbox_conf_{}_weight".format(layer_idx, branch_idx),
                lr_mult=1,
                wd_mult=1,
                init=mx.init.Xavier()
            )
            conf_bias = mx.sym.Variable(
                name="layer_{}_shared_mbox_conf_{}_bias".format(layer_idx, branch_idx),
                lr_mult=2,
                wd_mult=0,
                init=mx.init.Constant(0)
            )

            layer_mbox_weights.append((loc_weight, loc_bias, conf_weight, conf_bias))
        shared_weights.append(layer_mbox_weights)
    return shared_weights

def get_symbol_rolling_train(rolling_time,
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
    rolling_time: int
        rolling time
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

    assert len(sizes) == len(from_layers) or len(sizes) == (
        (len(from_layers) - 1) * rolling_time + 1)

    mbox_shared_weights = None
    if len(sizes) == ((len(from_layers) - 1) * rolling_time + 1):
        mbox_shared_weights = _get_multibox_shared_weights(len(layers), 4)
    # group output
    out = add_multibox_and_loss_for_extra(
        layers,
        label=label,
        num_classes=num_classes,
        num_filters=num_filters,
        sizes=sizes,
        ratios=ratios,
        normalizations=normalizations,
        steps=steps,
        nms_thresh=nms_thresh,
        force_suppress=force_suppress,
        nms_topk=nms_topk,
        rolling_idx=0,
        mbox_shared_weights=mbox_shared_weights)

    outputs = [out]

    # Rolling Layers
    last_rolling_layers = layers

    shared_weights = _get_shared_weights(len(last_rolling_layers), strides)

    for roll_idx in range(1, rolling_time + 1):
        roll_layers = create_rolling_struct(last_rolling_layers, kwargs["data_shape"], \
            num_filters=num_filters, strides=strides, pads=pads, rolling_rate=rolling_rate, \
            roll_idx=roll_idx, conv2=False, normalize=True, shared_weights=shared_weights)

        out = add_multibox_and_loss_for_extra(
            roll_layers,
            label=label,
            num_classes=num_classes,
            num_filters=num_filters,
            sizes=sizes,
            ratios=ratios,
            normalizations=normalizations,
            steps=steps,
            nms_thresh=nms_thresh,
            force_suppress=force_suppress,
            nms_topk=nms_topk,
            rolling_idx=roll_idx,
            mbox_shared_weights=mbox_shared_weights)

        outputs.append(out)

        last_rolling_layers = roll_layers

    return mx.symbol.Group(outputs)


def get_symbol_rolling_test(rolling_time,
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

    mbox_shared_weights = None
    if len(sizes) == ((len(from_layers) - 1) * rolling_time + 1):
        mbox_shared_weights = _get_multibox_shared_weights(len(layers), 4)

    if len(sizes) == (len(from_layers) - 1) * rolling_time + 1:
        loc_preds, cls_preds, anchor_boxes = branched_multibox_layer(layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, branch_num=4, shared_weights=mbox_shared_weights)
    elif len(sizes) == len(from_layers):
        loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    outputs = [out]

    shared_weights = _get_shared_weights(len(layers), strides)

    for roll_idx in range(1, rolling_time + 1):
        roll_layers = create_rolling_struct(
            layers,
            kwargs["data_shape"],
            num_filters=num_filters,
            strides=strides,
            pads=pads,
            rolling_rate=rolling_rate,
            roll_idx=roll_idx,
            conv2=False,
            normalize=True,
            shared_weights=shared_weights)
        out = add_multibox_for_extra(
            roll_layers,
            num_classes=num_classes,
            num_filters=num_filters,
            sizes=sizes,
            ratios=ratios,
            normalizations=normalizations,
            steps=steps,
            nms_thresh=nms_thresh,
            force_suppress=force_suppress,
            nms_topk=nms_topk,
            rolling_idx=roll_idx,
            mbox_shared_weights=mbox_shared_weights)

        outputs.append(out)

    return mx.sym.Group(outputs)
