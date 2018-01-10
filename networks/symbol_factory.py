# -*- coding: utf-8 -*-
"""Presets for various network configurations"""
from __future__ import absolute_import
import logging
import numpy as np
import networks.symbol_builder as symbol_builder
from networks.recurrent_rolling_net import get_symbol_rolling_train, get_symbol_rolling_test


def get_scales(min_scale=0.2,
               max_scale=0.9,
               num_layers=6,
               branch=False,
               branch_num=4):
    """
    Following the ssd arxiv paper, regarding the calculation of scales & ratios

    Parameters
    ----------
    min_scale: float
    max_scale: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list

    return
    ------
    sizes: list
        list of scale sizes per feature layer
    ratios: list
        list of anchor_ratios per feature layer
    """
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100 * min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    if branch:
        min_sizes_b = []
        max_sizes_b = []
        for i in range(0, len(min_sizes) - 1):
            for j in range(0, branch_num):
                min_sizes_b.append(min_sizes[i] + j * (
                    min_sizes[i + 1] - min_sizes[i]) / branch_num)
        min_sizes_b.append(min_sizes[-1])

        for i in range(0, len(max_sizes) - 1):
            for j in range(1, branch_num + 1):
                max_sizes_b.append(min_sizes_b[branch_num * i + j])
        max_sizes_b.append(max_sizes[-1])

        min_sizes, max_sizes = min_sizes_b, max_sizes_b

    nums = num_layers if not branch else len(min_sizes)
    scales = []
    for layer_idx in range(nums):
        scales.append([
            min_sizes[layer_idx],
            np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))
        ])

    return scales


def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, 1024, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 2]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = get_scales(
                min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [
                x / 512.0 for x in [8, 16, 32, 64, 128, 256, 512]
            ]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, 1024, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = get_scales(
                min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [
                x / 300.0 for x in [8, 16, 32, 64, 100, 300]
            ]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' %
                         data_shape)
        return locals()
    elif network == 'inceptionv3':
        if data_shape >= 448:
            from_layers = [
                'ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat',
                '', '', '', ''
            ]
            num_filters = [768, 2048, 512, 256, 256, 128]
            strides = [-1, -1, 2, 2, 2, 2]
            pads = [-1, -1, 1, 1, 1, 1]
            sizes = get_scales(
                min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        else:
            from_layers = [
                'ch_concat_mixed_2_chconcat', 'ch_concat_mixed_7_chconcat',
                'ch_concat_mixed_10_chconcat', '', '', ''
            ]
            num_filters = [288, 768, 2048, 256, 256, 128]
            strides = [-1, -1, -1, 2, 2, 2]
            pads = [-1, -1, -1, 1, 1, 1]
            sizes = get_scales(
                min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [1024, 2048, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == "resnet50-rolling":
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = "resnet"
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [256, 256, 256, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == "resnet50-rb":
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = "resnet"
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [256, 256, 256, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.15,
            max_scale=0.85,
            num_layers=len(from_layers),
            branch=True,
            branch_num=4)
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '', '', '']
        num_filters = [1024, 1024, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobilenet':
        from_layers = ['conv_12_relu', 'conv_14_relu', '', '', '', '', '']
        num_filters = [512, 512, 512, 256, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet121':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 24, 16]
        num_stage = 4
        growth_rate = 32
        bottle_neck = True
        from_layers = [
            'DBstage3_concat24', 'DBstage4_concat16', '', '', '', ''
        ]
        num_filters = [1024, 1024, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet-tiny':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 18, 12]
        num_stage = 4
        growth_rate = 16
        bottle_neck = True
        from_layers = [
            'DBstage2_concat12', 'DBstage3_concat18', '', '', '', ''
        ]
        num_filters = [256, 416, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(
            min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network,
                                                                    data_shape)
        raise NotImplementedError(msg)


def get_symbol_train(network,
                     data_shape,
                     rolling=False,
                     rolling_time=4,
                     **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)

    if rolling:
        return get_symbol_rolling_train(rolling_time, **config)
    else:
        return symbol_builder.get_symbol_train(**config)


def get_symbol(network, data_shape, rolling=False, rolling_time=4, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)

    if rolling:
        return get_symbol_rolling_test(rolling_time, **config)
    else:
        return symbol_builder.get_symbol(**config)
