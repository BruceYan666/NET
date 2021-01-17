import torch
import argparse
import pdb
import numpy as np
from model import *
from dataset.data import CiFar10Dataset,DataPreProcess
from mmcv import Config
from torch.utils.data import DataLoader
from log.logger import Logger
from utils import get_network

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Loading pretrain data')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, required=True, help='input which model to use')
    args = parser.parse_args()
    return args

def dataLoad (cfg):
    test_data = CiFar10Dataset(txt = cfg.PARA.data.test_data_txt, transform='for_test')
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.PARA.test.BATCH_SIZE, drop_last=True, shuffle=False, num_workers= cfg.PARA.train.num_workers)
    return test_loader

# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):#将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
        conf_matrix[t, p] += 1
    return conf_matrix

def test(net, epoch, test_loader, log, args, cfg):
    conf_matrix = torch.zeros(cfg.PARA.test.NUM_CLASSES, cfg.PARA.test.NUM_CLASSES)
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1]
            labels = torch.max(labels, 1)[1]
            conf_matrix = confusion_matrix(predicted, labels, conf_matrix) #混淆矩阵
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print(conf_matrix)
        precision = []
        recall = []
        row_sum = torch.sum(conf_matrix, dim=1) #每一行的和
        col_sum = torch.sum(conf_matrix, dim=0) #每一列的和
        for i in range(cfg.PARA.test.NUM_CLASSES):
            precision.append(conf_matrix[i][i] / col_sum[i])
            recall.append(conf_matrix[i][i] / row_sum[i])
        prec_aver = np.mean(precision)
        recall_aver = np.mean(recall)
        F1 = 2 * prec_aver * recall_aver / (prec_aver + recall_aver)
        log.logger.info('accuracy=%.3f%%, precision=%.3f%%, recall=%.3f%%, F1=%.3f' % (100 * correct / total, 100 * prec_aver, 100 * recall_aver, F1))
        f = open("./cache/visual/"+args.net+"_test.txt", "a")
        f.write("epoch=%d,acc=%.3f" % (epoch, correct / total))
        f.write('\n')
        f.close()
def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('./cache/log/' + args.net + '_testlog.txt', level='info')
    log.logger.info('==> Preparing data <==')
    test_loader = dataLoad(cfg)
    log.logger.info('==> Loading model <==')
    net = get_network(args).cuda()
    net = torch.nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    log.logger.info("==> Waiting Test <==")
    # for epoch in range(1, cfg.PARA.train.EPOCH+1):
    epoch = 121
    checkpoint = torch.load('./cache/checkpoint/'+args.net+'/'+ str(epoch) +'ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    test(net, epoch, test_loader, log, args, cfg)

if __name__ == '__main__':
    main()
