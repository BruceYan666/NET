import torch
import argparse
from model import *
from dataset.data import CiFar10Dataset,DataPreProcess
from mmcv import Config
from torch.utils.data import DataLoader
from log.logger import Logger
from utils import get_network

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, required=True, help='input which model to use')
    args = parser.parse_args()
    return args

def dataLoad (cfg):
    test_data = CiFar10Dataset(txt = cfg.PARA.data.test_data_txt, transform='for_test')
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.PARA.test.BATCH_SIZE, drop_last=True, shuffle=False, num_workers= cfg.PARA.train.num_workers)
    return test_loader

def test(epoch, test_loader, log):
    log.logger.info("==> Waiting Test <==")
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        log.logger.info('测试分类准确率为：%.3f%%' % (100 * correct / total))
        f = open("./cache/visual/"+args.net+"_test.txt", "a")
        f.write("epoch=%d,acc=%.3f%%" % (epoch , 100. * correct / total))
        f.write('\n')
        f.close()
def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('./cache/log/' + args.net + '_testlog.txt', level='info')
    log.logger.info('==> Preparing data <==')
    test_loader = dataLoad(cfg)
    log.logger.info('==> Loading model <==')
    global net
    net = get_network(args).cuda()
    net = torch.nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    for epoch in (1, cfg.PARA.train.EPOCH+1):
        checkpoint = torch.load('./checkpoint/'+args.net+'/'+ epoch +'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(epoch, test_loader, log)

if __name__ == '__main__':
    main()
