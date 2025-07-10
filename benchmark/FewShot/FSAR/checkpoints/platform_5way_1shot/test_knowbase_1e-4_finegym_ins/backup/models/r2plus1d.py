import torch
import torch.nn as nn

from torchvision.models.video.resnet import (
    BasicBlock, Bottleneck, R2Plus1dStem, _video_resnet)

__all__ = ['r2plus1d_18', 'r2plus1d_34', 'r2plus1d_152']


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
                in_planes * 3 * 3 + 3 * out_planes)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)



def r2plus1d_18(pretrained=False, progress=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    return _video_resnet('r2plus1d_18',
                         False, False,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)


def r2plus1d_34(pretrained=False, progress=False, **kwargs):
    return _video_resnet('r2plus1d_34',
                         False, False,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[3, 4, 6, 3],
                         stem=R2Plus1dStem, **kwargs)


def r2plus1d_152(pretrained=False, progress=False, **kwargs):
    return _video_resnet('r2plus1d_152',
                         False, False,
                         block=Bottleneck,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[3, 8, 36, 3],
                         stem=R2Plus1dStem, **kwargs)


def get_fine_tuning_parameters_layer_lr(model, ft_begin_index, layer_lr):
    # ft_begin_index is 0 by default
    ft_module_names = []
    layer_lr_dict = {}
    count = 0
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
        layer_lr_dict['layer{}'.format(i)] = layer_lr[count]
        count = count + 1
    ft_module_names.append('fc')
    layer_lr_dict['fc'] = layer_lr[count]

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v, 'lr': layer_lr_dict[ft_module]})
                break
        else:
            if ft_begin_index == 0:
                print(f'{k} uses same lr as layer0')
                parameters.append({'params': v, 'lr': layer_lr_dict['layer0']})
            else:
                parameters.append({'params': v, 'lr': 0.0})
    
    return parameters

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
       return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

