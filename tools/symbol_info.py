import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks.symbol_factory import get_config
from networks.symbol_builder import get_symbol_train

def make_net(network, data_shape):
    kwargs = {
        "num_classes": 20,
        "nms_thresh": 0.45,
        "force_suppress": False,
        "nms_topk": 400,
        "minimum_negative_samples": 0
    }

    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    assert network.startswith(config["network"])
    assert data_shape == config["data_shape"]
    net = get_symbol_train(**config)
    return net

def output_shape_of_layers(net, layers, input_shape):

    net = make_net(net, input_shape)

    internals = net.get_internals()

    res = []
    for layer in layers:
        layer_sym = internals[layer + "_output"]
        _, output_shapes, _ = layer_sym.infer_shape(
            data=(32, 3, input_shape, input_shape))
        res.append(output_shapes[0])
    return res

def output_info(net_infos):
    res = []
    for net in net_infos:
        output_shapes = output_shape_of_layers(
            net["network"], net["from_layers"], net["data_shape"])
        net["output_shapes"] = output_shapes
        res.append(net)
    return res

def main():
    net_infos = [
        {
            "network": "vgg16_reduced",
            "data_shape": 300,
            "from_layers": ['relu4_3', 'relu7']
        },
        {
            "network": "vgg16_reduced",
            "data_shape": 512,
            "from_layers": ['relu4_3', 'relu7']
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
            "data_shape": 512,
            "from_layers": ['_plus12', '_plus15']
        },
        # {
        #     "network": "resnet101",
        #     "data_shape": 512,
        #     "from_layers": ['_plus12', '_plus15']
        # },
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

    res = output_info(net_infos)
    print(res)
    return res


if __name__ == '__main__':
    main()
