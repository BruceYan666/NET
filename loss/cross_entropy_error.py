import torch
import torch.nn as nn
import pdb

class CrossEntropyloss(nn.Module):

    def __init__(self):
        super(CrossEntropyloss, self).__init__()

    def forward(self,output,target):
        loss = nn.CrossEntropyLoss()(output, target)
        return loss

def test():
    torch.manual_seed(2020)
    output=torch.randn(1,3)
    target=torch.tensor([2])
    loss=CrossEntropyloss()
    y=loss(output,target)
    print(y)

def main():
    test()

if __name__ == '__main__':
    main()
