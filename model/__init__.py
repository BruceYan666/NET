'''
__init__.py 文件的作用是将文件夹变为一个Python模块
__all__ 把注册在__init__.py文件中__all__列表中的模块和包导入到当前文件中
'''

from .ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from .InceptionV4 import inceptionv4, inception_resnet_v2
from .SqueezeNet import squeezenet

__all__ = [
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'inceptionv4',
    'inception_resnet_v2', 'squeezenet'
]