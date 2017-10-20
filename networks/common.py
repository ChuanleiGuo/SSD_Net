import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvActLayer(nn.Module):
    """
    wrapper for a small Convolution group

    ## Parameters:

    from_layer: nn.Module
        continue on which layer
    num_filter: int
        how many filters to use in conv layer
    kernel: tuple (int, int)
        kernel size (h, w)
    padding: tuple (int, int)
        padding size (h, w)
    stride: tuple (int, int)
        stride size (h, w)
    act_type: str
        activation function, default to be relu
    use_batchnorm: bool
        whether to use batch normalization
    
    ## Returns:

    (conv, relu) -> result
    """

    def __init__(self, from_layer, num_filter, kernel=(1, 1),
                 padding=(0, 0), stride=(1, 1), act_type="relu",
                 use_batchnorm=False):
        super(ConvActLayer, self).__init__()
        self.from_layer = from_layer
        # TODO: `lt_mult` for bias of conv
        self.conv = nn.Conv2d(
            from_layer.out_channels,
            num_filter, kernel, stride, padding, bias=True)
        if use_batchnorm:
            self.use_batchnorm = True
            self.batchnorm = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.from_layer(x)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.relu(x)
        return x
