import os
import sys
from model import *
def get_network(args):
    """ return given network
    """
    if args.net == 'vgg16':
        net = vgg16(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg13':
        net = vgg13(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg11':
        net = vgg11(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg19':
        net = vgg19(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg16_bn':
        net = vgg16_bn(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg13_bn':
        net = vgg13_bn(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg11_bn':
        net = vgg11_bn(pretrained=args.pretrain).cuda()
    elif args.net == 'vgg19_bn':
        net = vgg19_bn(pretrained=args.pretrain).cuda()
    elif args.net == 'inceptionv4':
        net = inceptionv4().cuda()
    elif args.net == 'inceptionresnetv2':
        net = inception_resnet_v2().cuda()
    elif args.net == 'ResNet18':
        net = ResNet18(pretrained=args.pretrain).cuda()
    elif args.net == 'ResNet34':
        net = ResNet34(pretrained=args.pretrain).cuda()
    elif args.net == 'ResNet50':
        net = ResNet50(pretrained=args.pretrain).cuda()
    elif args.net == 'ResNet101':
        net = ResNet101(pretrained=args.pretrain).cuda()
    elif args.net == 'ResNet152':
        net = ResNet152(pretrained=args.pretrain).cuda()
    elif args.net == 'squeezenet':
        net = squeezenet().cuda()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
