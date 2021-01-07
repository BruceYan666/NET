import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def softmax(self, x, axis=1): #axis=1表示对行处理，axis=0表示对列处理
        x_max, _ = x.max(axis=axis, keepdims=True) #每一行的最大值
        # 当 keepidms=True,保持其二维或者三维的特性(结果保持其原来维数)，默认为False,不保持其二维或者三维的特性(结果不保持其原来维数)
        x = x - x_max  #每一行减去每一行的最大值
        y = torch.exp(x)
        return y / y.sum(axis=axis, keepdims=True) #每一行的和

    def forward(self, x, labels):
        batch_size = x.shape[0]
        y = self.softmax(x)
        # y = F.softmax(x, dim=1)
        return -torch.sum(labels * torch.log(y + 1e-7)) / batch_size

def main():
    t1 = torch.from_numpy(np.array([2]))
    t = torch.from_numpy(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
    x = torch.randn(1, 10)
    a = CrossEntropy()
    loss = a(x, t)
    b = nn.CrossEntropyLoss()  #自带onehot编码和softmax激活函数
    loss1 = b(x, t1)
    print(loss, loss1)

if __name__ == '__main__':
    main()

