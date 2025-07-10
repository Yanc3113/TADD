import math

import torch
import torchvision
from torch.nn import ReplicationPad3d

from models import inflate


def i3res_50(sample_size=112, sample_duration=16, num_classes=400):
    resnet = torchvision.models.resnet50(pretrained=True)
    return I3ResNet(resnet, Bottleneck3d, expansion=4, sample_size=sample_size, frame_nb=sample_duration, class_nb=num_classes)

def i3res_34(sample_size=112, sample_duration=16, num_classes=400):
    resnet = torchvision.models.resnet34(pretrained=True)
    return I3ResNet(resnet, BasicBlock3d, expansion=1, sample_size=sample_size, frame_nb=sample_duration, class_nb=num_classes)



class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, block_func, expansion=4, sample_size=112, frame_nb=16, class_nb=1000, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1, block_func)
        self.layer2 = inflate_reslayer(resnet2d.layer2, block_func)
        self.layer3 = inflate_reslayer(resnet2d.layer3, block_func)
        self.layer4 = inflate_reslayer(resnet2d.layer4, block_func)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            last_duration = int(math.ceil(frame_nb / 16))
            last_size = int(math.ceil(sample_size / 32))

            self.avgpool = torch.nn.AdaptiveAvgPool3d((1,1, 1))
            # self.avgpool = torch.nn.AvgPool3d(
            #     (last_duration, last_size, last_size), stride=1)
            self.fc = torch.nn.Linear(512*expansion, class_nb)

            #self.avgpool = inflate.inflate_pool(
            #    resnet2d.avgpool, time_dim=final_time_dim)
            #self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            #print(x.size())

            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


def inflate_reslayer(reslayer2d, block_func):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = block_func(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class BasicBlock3d(torch.nn.Module):
    def __init__(self, basicblock2d):
        super(BasicBlock3d, self).__init__()

        spatial_stride = basicblock2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(
            basicblock2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(basicblock2d.bn1)

        self.conv2 = inflate.inflate_conv(
            basicblock2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate.inflate_batch_norm(basicblock2d.bn2)

        self.relu = torch.nn.ReLU(inplace=True)

        if basicblock2d.downsample is not None:
            self.downsample = inflate_downsample(
                basicblock2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = basicblock2d.stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d
