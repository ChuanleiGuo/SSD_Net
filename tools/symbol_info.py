import os
import sys
import math
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks.symbol_factory import get_config
from networks.symbol_builder import get_symbol_train
from networks.recurrent_rolling_net import get_symbol_rolling_train

def make_net(network, data_shape, rolling=False, rolling_time=4):
    kwargs = {
        "num_classes": 20,
        "nms_thresh": 0.45,
        "force_suppress": False,
        "nms_topk": 400,
        "minimum_negative_samples": 0
    }
    config = get_config(network, data_shape[2], **kwargs).copy()
    config.update(kwargs)
    assert network.startswith(config["network"])
    assert data_shape[2] == config["data_shape"]
    if rolling:
        net = get_symbol_rolling_train(rolling_time, **config)
    else:
        net = get_symbol_train(**config)
    return net

def output_shape_of_layers(net, layers, input_shape, rolling=False):

    if isinstance(input_shape, int):
        input_shape = (32, 3, input_shape, input_shape)
    elif isinstance(input_shape, tuple):
        input_shape = (32, 3, input_shape[0], input_shape[1])

    net = make_net(net, input_shape, rolling=rolling, rolling_time=4)

    internals = net.get_internals()

    res = []
    for layer in layers:
        layer_sym = internals[layer + "_output"]
        _, output_shapes, _ = layer_sym.infer_shape(
            data=input_shape)
        res.append(output_shapes[0])
    return res

def output_info(net_infos, rolling=False):
    res = []
    for net in net_infos:
        output_shapes = output_shape_of_layers(
            net["network"], net["from_layers"], net["data_shape"], rolling=rolling)
        net["output_shapes"] = output_shapes
        res.append(net)
    return res

def main():
    net_infos = [
        {
            "network": "vgg16_reduced",
            "data_shape": 300,
            "from_layers": ['relu4_3', 'relu7'] + ["multi_feat_%d_conv_3x3_relu" % k for k in range(2, 6)]
        },
        {
            "network": "vgg16_reduced",
            "data_shape": (2560, 768),
            "from_layers": ['relu4_3', 'relu7'] + ["multi_feat_%d_conv_3x3_relu" % k for k in range(2, 7)]
        },
        {
            "network": "inceptionv3",
            "data_shape": 300,
            "from_layers": ['ch_concat_mixed_2_chconcat', 'ch_concat_mixed_7_chconcat', "ch_concat_mixed_10_chconcat"]
        },
        {
            "network": "inceptionv3",
            "data_shape": 512,
            "from_layers": ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat']
        },
        {
            "network": "resnet50",
            "data_shape": (2560, 768),
            "from_layers": ['_plus12', '_plus15'] + ["multi_feat_%d_conv_3x3_relu" % k for k in range(2, 6)]
        },
        {
            "network": "resnet101",
            "data_shape": 512,
            "from_layers": ['_plus29', '_plus32'] + ["multi_feat_%d_conv_3x3_relu" % k for k in range(2, 6)]
        },
        {
            "network": "densenet121",
            "data_shape": 512,
            "from_layers": ['DBstage3_concat24', 'DBstage4_concat16']
        },
        {
            "network": "densenet-tiny",
            "data_shape": 512,
            "from_layers": ['DBstage2_concat12', 'DBstage3_concat18']
        },
    ]

    # res = output_info(net_infos)
    # pprint(res)

    print('-' * 10  + "Rolling" + '-' * 10)
    net_infos = [
        # {
        #     "network": "resnet50-rolling",
        #     "data_shape": (2560, 768),
        #     "from_layers": ["multi_feat_%d_conv_3x3_relu" % k for k in range(0, 6)]
        # },
        {
            "network": "resnet50-rb",
            "data_shape": (2560, 768),
            "from_layers": ["multi_feat_%d_conv_3x3_relu" % k for k in range(0, 6)]
        },
    ]
    res = output_info(net_infos, rolling=True)
    pprint(res)


def prior_sizes(width, height, min_ratio, max_ratio, source_layers):
    min_dim = min(width, height)
    step = int(math.floor((max_ratio - min_ratio) / (len(source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 6.7 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    return min_sizes, max_sizes

if __name__ == '__main__':
    main()
