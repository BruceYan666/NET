import pdb
import torch
import numpy as np

# def ToOnehot(labels,num_classes):
#     temp = torch.eyes(len(labels),num_classes)
#     for i in range(len(labels)):
#         temp[i,labels[i]] = 1
#     return temp

def ToOnehot(label, num_classes):
    temp = torch.eye(num_classes)
    return temp[label]

def ToOnehots(labels,num_classes):
    one_hot = torch.zeros([labels.shape[0], num_classes])
    # pdb.set_trace()
    for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1
    return one_hot



def test():
    labels=torch.tensor([0,5,6])
    label = torch.tensor(5)
    a = ToOnehot(label, 10)
    b = ToOnehots(labels,10)
    # pdb.set_trace()
    print(a)
    print(b)

if __name__=="__main__":
    test()