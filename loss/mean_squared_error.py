import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def softmax(self, x, axis=1):
        x_max, _ = x.max(axis=axis, keepdims=True)
        x = x - x_max
        y = torch.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def forward(self, x, t):
        y = self.softmax(x)
        # y = F.softmax(x, dim=1) #按行进行softmax正规化
        m, n = y.size()
        length = m * n
        return torch.sum((y - t)**2) / length

def main():
    t = torch.Tensor([[0, 0, 1], [0, 1, 0]])
    x = torch.Tensor([[1, 2, 3], [3, 2, 1]])
    a = MeanSquaredError()
    b = nn.MSELoss()#没有onehot编码和softmax激活函数
    loss = a(x, t)
    x1 = F.softmax(x, dim=1)
    loss1 = b(x1, t)
    print(loss, loss1)

if __name__ == '__main__':
    main()




