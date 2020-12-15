from __future__ import print_function
import torch
import os
import torchvision
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.optim as optim
import torch.nn as nn
import argparse
import pdb
import mmcv
import logging
import importlib
from model import *
from loss import *
from dataset.data import CiFar10Dataset,DataPreProcess
from log.logger import Logger
from torch.utils.data import DataLoader
from mmcv import Config
from mmcv.runner import load_checkpoint
from utils import get_network

#logging.basicConfig(filename='TrainLog.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#如果在logging.basicConfig()设置filename 和filemode，则只会保存log到文件，不会输出到控制台
#logging.disable(logging.CRITICAL)
assert torch.cuda.is_available(), 'Error: CUDA not found!'

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, required=True, help='input which model to use')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Loading pretrain data')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epoch', '-e', default=None, help='resume from epoch')
    args = parser.parse_args()
    return args

def dataLoad (cfg):
    train_data = CiFar10Dataset(txt = cfg.PARA.data.train_data_txt, transform='for_train')
    val_data = CiFar10Dataset(txt = cfg.PARA.data.val_data_txt, transform='for_val')
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.PARA.train.BATCH_SIZE, drop_last=True, shuffle=True, num_workers= cfg.PARA.train.num_workers)
    val_loader = DataLoader(dataset=val_data, batch_size=cfg.PARA.train.BATCH_SIZE, drop_last=True, shuffle=False, num_workers= cfg.PARA.train.num_workers)
    return train_loader , val_loader


def train (epoch, train_loader, cfg, net, args, log):
    criterion = CrossEntropyloss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=cfg.PARA.train.LR, momentum=cfg.PARA.train.momentum, weight_decay=cfg.PARA.train.wd)
    log.logger.info('Epoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(train_loader, 0):
        length = len(train_loader)
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        log.logger.info('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
    f = open("./cache/visual/"+args.net+"_train.txt", "a")
    f.write("epoch=%d,acc=%.3f,loss=%.03f" % (epoch + 1, correct / total, sum_loss / length))
    f.write('\n')
    f.close()
    log.logger.info('Saving Model')
    state = {
        'net': net.state_dict(),
        'epoch': epoch
    }
    if not os.path.isdir('./cache/checkpoint/'+args.net):
        os.mkdir('./cache/checkpoint/'+args.net)
    torch.save(state, './cache/checkpoint/'+args.net+'/'+str(epoch+1)+ 'ckpt.pth')

def validate(epoch, val_loader, net, args, log):
    log.logger.info('Waiting Validation')
    with torch.no_grad():#强制之后的内容不进行计算图构建,不用梯度反传
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(val_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        log.logger.info('测试分类准确率为：%.3f%%' % (100 * correct / total))

def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('./cache/log/' + args.net + '_trainlog.txt', level='info')
    log.logger.info('Preparing data')
    train_loader , val_loader = dataLoad(cfg)
    start_epoch = 0
    if args.pretrain:
        log.logger.info('Loading Pretrain Data')
    net = get_network(args).cuda()
    net = torch.nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    torch.backends.cudnn.benchmark = True
    if args.resume:
        log.logger.info('Resuming from checkpoint')
        weighted_file = os.path.join('./checkpoint/'+args.net, args.epoch + 'ckpt.pth')
        checkpoint = torch.load(weighted_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
    for epoch in range(start_epoch, cfg.PARA.train.EPOCH):
        train(epoch, train_loader, cfg, net, args, log)
        validate(epoch, val_loader, net, args, log)
    log.logger.info("Training Finished, Total EPOCH=%d" % cfg.PARA.train.EPOCH)

if __name__ == '__main__':
    main()
