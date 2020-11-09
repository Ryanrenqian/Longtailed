
import torch
import torch.nn.init as init
import torch.nn as nn
from utils import *
from os import path
import math


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class SparseClassifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048,num_head=2,eps=0.03125):
        super(SparseClassifier, self).__init__()
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.head_class = num_classes//num_head
        self.eps = eps
        self.weights = [nn.Parameter(torch.Tensor(self.head_class, self.head_dim).cuda(), requires_grad=True) for i in range(num_head)]
        for weight in self.weights:
            self.reset_parameters(weight)
    
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, *args):
        output = []
        x_list = torch.split(x,self.head_dim,dim=1)
        for x,weight in zip(x_list,self.weights):
            x = self.l2_norm(x)
            fc = self.l2_norm(weight,self.eps)
            output.append(torch.mm(x,fc.T))
        output = torch.cat(output, dim=1)
        return output,None
      
    def l2_norm(self, x, weight=None):
        if weight:
            return  x / (torch.norm(x, 2, 1, keepdim=True)+weight)
        return x / torch.norm(x, 2, 1, keepdim=True)
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, eps=0.03125):
    print('Loading SparseClassifier Classifier.')
    clf = SparseClassifier(num_classes, feat_dim, num_head=num_head,eps=eps)
    return clf

if __name__ == '__main__':
    x = torch.randn((10,2048)).cuda()
    y = torch.randint(0,1000,(10,1)).squeeze(1).cuda()
    print(y)
    classifer = DotProduct_Classifier()
    target,_ = classifer(x)
    loss = nn.CrossEntropyLoss()(target,y)
    print(loss)
    loss.backward()
    print(classifer.weight.grad.data.size())
    print(classifer.weight.grad.data.norm(2,1).size())
