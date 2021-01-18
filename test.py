import torch
import argparse
import pdb
import numpy as np
import itertools
import matplotlib.pyplot as plt
from model import *
from dataset.data import CiFar10Dataset,DataPreProcess
from mmcv import Config
from torch.utils.data import DataLoader
from log.logger import Logger
from utils import get_network
from itertools import cycle

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Loading pretrain data')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, default='ResNet18', help='input which model to use')
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


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def roc_curve(label_array, score_array):
    num = len(score_array)
    num_true = np.sum(label_array == 1)
    fpr = []
    tpr = []
    for T in range(0, 100, 2):
        T = T / 100
        mask = (score_array >= T)
        TP = np.sum(label_array[mask] == 1)
        FP = np.sum(label_array[mask] == 0)
        FN = num_true - TP
        TN = num - num_true - FP
        fpr.append(FP / (FP + TN))
        tpr.append(TP / (TP + FN))
    return fpr, tpr

def evaluate(conf_matrix, NUM_CLASSES):
    precision = []
    recall = []
    row_sum = torch.sum(conf_matrix, dim=1)  # 每一行的和
    col_sum = torch.sum(conf_matrix, dim=0)  # 每一列的和
    for i in range(NUM_CLASSES):
        precision.append(conf_matrix[i][i] / col_sum[i])
        recall.append(conf_matrix[i][i] / row_sum[i])
    prec_aver = np.mean(precision)
    recall_aver = np.mean(recall)
    F1 = 2 * prec_aver * recall_aver / (prec_aver + recall_aver)
    return prec_aver, recall_aver, F1

def test(net, epoch, test_loader, log, args, cfg):
    conf_matrix = torch.zeros(cfg.PARA.test.NUM_CLASSES, cfg.PARA.test.NUM_CLASSES)
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        score_list = []  # 存储预测得分
        label_list = []  # 存储真实标签
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            score_list.extend(outputs.detach().cpu().numpy()) #list.extend()把一个序列seq的内容在尾部逐一加入
            label_list.extend(labels.data.cpu().numpy()) #list.append() 将整个对象在列表末尾加入
            predicted = torch.max(outputs.data, 1)[1]
            labels = torch.max(labels, 1)[1]
            conf_matrix = confusion_matrix(predicted, labels, conf_matrix) #混淆矩阵
            total += labels.size(0)
            correct += (predicted == labels).sum()

        attack_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False, title='Normalized confusion matrix')
        prec_aver, recall_aver, F1 = evaluate(conf_matrix, cfg.PARA.test.NUM_CLASSES)
        log.logger.info('accuracy=%.3f%%, precision=%.3f%%, recall=%.3f%%, F1=%.3f' % (100 * correct / total, 100 * prec_aver, 100 * recall_aver, F1))

        score_array = np.array(score_list)
        label_array = np.array(label_list)
        fpr_dict = dict()
        tpr_dict = dict()
        # 每一类的roc曲线
        for i in range(cfg.PARA.test.NUM_CLASSES):
            fpr_dict[i], tpr_dict[i] = roc_curve(label_array[:, i], score_array[:, i])
        # 平均roc曲线
        fpr_dict["micro"], tpr_dict["micro"] = roc_curve(label_array.ravel(), score_array.ravel()) #.ravel()将多维数组转换为一维数组
        plt.figure(1)
        lw = 2    #linewidth 线宽为2个像素值
        plt.plot(fpr_dict["micro"], tpr_dict["micro"], label='average ROC curve ', color='gold', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'beige', 'khaki', 'cornflowerblue'])  #cycle()函数重复循环一组值
        for i, color in zip(range(cfg.PARA.test.NUM_CLASSES), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw, label='ROC curve of class {0} '.format(i))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # y=x
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right", fontsize='x-small')
        plt.show()


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
