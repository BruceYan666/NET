import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from mmcv.runner import load_checkpoint
import pdb
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        self.linear = nn.Linear(512, 512)

        self.classifier = nn.Sequential(
            # nn.Linear(512, 512),#cifar10输入图片尺寸为32x32，不是224x224
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.features(x) #输出x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)
        x = x.view(x.size(0), -1)#将前面多维度的tensor展平成一维，x.size(0)是batchsize，指转换后有几行。-1是自适应的意思，指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        # 比如原来的数据一共12个，batchsize为2，就会view成2*6，batchsize为4，就会就会view成4*3
        # pdb.set_trace()
        x = self.linear(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():#self.modules()类内初始化，会遍历model中所有的子层
            if isinstance(m, nn.Conv2d):#isinstance(object, classinfo)，如果对象（object）的类型与classinfo相同则返回True，否则返回False
                # pdb.set_trace()
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg11'])
    return model


def vgg11_bn(pretrained=False, **kwargs):
    # pdb.set_trace()
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg11_bn'])
    return model


def vgg13(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg13'])
    return model


def vgg13_bn(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg13_bn'])
    return model


def vgg16(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg16'])
    return model


def vgg16_bn(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg16_bn'])
    return model


def vgg19(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg19'])
    return model


def vgg19_bn(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        load_checkpoint(model,model_urls['vgg19_bn'])
    return model


def test():
    net = vgg19()
    print(net)

def main():
    test()

if __name__ == '__main__':
    main()