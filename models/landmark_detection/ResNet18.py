"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch 
based on torchvision.models.resnet.
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, model_urls
import copy

class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)

class Upsampling(nn.Sequential):
    """
    3-layers deconvolution used in `Simple Baseline <https://arxiv.org/abs/1804.06208>`_.
    """
    def __init__(self, in_channel=2048, hidden_dims=(200, 200, 200), kernel_sizes=(4, 4, 4), bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class PoseResNet(nn.Module):
    """
    `Simple Baseline <https://arxiv.org/abs/1804.06208>`_ for keypoint detection.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsampling(x)
        x = self.head(x)
        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
        ]

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def pose_resnet18(image_size, num_keypoints, pretrained_backbone, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    backbone = resnet18(pretrained_backbone, progress)
    upsampling = Upsampling(backbone.out_features, hidden_dims=(image_size, image_size, image_size), bias=deconv_with_bias)
    model = PoseResNet(backbone, upsampling, image_size, num_keypoints, finetune)
    return model