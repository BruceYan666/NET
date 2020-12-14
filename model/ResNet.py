import pdb
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from mmcv.runner import load_checkpoint

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']  #？？？？

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 输出的通道数由64变为256,所以下面要乘self.expansion
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,init_weights=True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
 
        self.inplanes = 64 #一开始先使用64*7*7的卷积核
        self.dilation = 1
        if replace_stride_with_dilation is None:#空洞卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))#str.format()格式化函数，是通过{}来代替以前的%
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) #3通道的输入RGB图像数据变为64通道的数据
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #在每个残差分支中初始化最后一个BN，以便残差分支以零开始，并且每个残差块的行为类似于一个恒等式
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0) #Bottleneck的最后一个BN是m.bn3
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) #BasicBlock的最后一个BN是m.bn2
        if init_weights:
            self._init_weight()


    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # kaiming高斯初始化，目的是使得Conv2d卷积层反向传播的输出的方差都为1
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 初始化m.weight，即gamma的值为1；m.bias即beta的值为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:#空洞卷积
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )#如果该层的输入的channel数inplanes和其输出的channel数planes * block.expansion不同（layers转换时）
             #那要使用1*1的卷积核将输入x低维转成高维，然后才能进行相加
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion#进行不同layers间输入输出通道数转换

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)



 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
 
        return x

def ResNet18(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ResNet(BasicBlock,[2, 2, 2, 2], **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['resnet18'])
    return model

def ResNet34(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ResNet(BasicBlock,[3, 4, 6, 3], **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['resnet34'])
    return model

def ResNet50(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ResNet(Bottleneck,[3, 4, 6, 3], **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['resnet50'])
    return model

def ResNet101(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ResNet(Bottleneck,[3, 4, 23, 3], **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['resnet101'])
    return model

def ResNet152(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ResNet(Bottleneck,[3, 8, 36, 3], **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['resnet152'])
    return model

def test():
    resnet = ResNet(BasicBlock,[2,2,2,2])
    print(resnet)
    x = torch.randint(1,256,(1,3,224,224))
    x = x.float()
    y = resnet(x)
    print(y)



def main():
    test()


if __name__ == '__main__':
    main()
