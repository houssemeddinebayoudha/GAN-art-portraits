import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
    return nn.Sequential(
        nn.Conv2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2, inplace=False))

def Deconv(n_input, n_output, k_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride, padding=padding,
            bias=False),
        nn.BatchNorm2d(n_output),
        nn.ReLU(inplace=True))

class Generator(nn.Module):
    def __init__(self, z=100, nc=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            Deconv(z, nc*8, 4,1,0),
            Deconv(nc*8, nc*4, 4,2,1),
            Deconv(nc*4, nc*2, 4,2,1),
            Deconv(nc*2, nc, 4,2,1),
            nn.ConvTranspose2d(nc,3, 4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self, nc=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                3, nc,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(nc, nc*2, 4,2,1),
            Conv(nc*2, nc*4, 4,2,1),
            Conv(nc*4, nc*8, 4,2,1),
            nn.Conv2d(nc*8, 1,4,1,0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        return self.net(input)

def normalize_tensor(tensor, normalize=True, value_range=None):
    """
    Normalize a tensor to a given value range (min, max). If normalize is False, the tensor is returned as is.

    Args:
        tensor (torch.Tensor): The input tensor to normalize.
        normalize (bool): Whether to normalize the tensor. Default is True.
        value_range (tuple or None): A tuple (min, max) specifying the range to normalize the tensor. 
                                     If None, the range is automatically computed from the tensor values.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if normalize:
        tensor = tensor.clone()  # Avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)  # Clamp the values to the provided range
            img.sub_(low).div_(max(high - low, 1e-5))  # Normalize the image

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))  # Use the min/max of the tensor

        norm_range(tensor, value_range)

    return tensor
