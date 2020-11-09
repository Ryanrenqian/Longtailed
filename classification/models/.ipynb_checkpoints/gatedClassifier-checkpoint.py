
import torch
import torch.nn.init as init
import torch.nn as nn
# from utils import *
from os import path
import math


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Gated(nn.Module):
    def __init__(self, head_class=50):
        super(Gated, self).__init__()
        self.linear = nn.Linear(head_class, head_class, bias=False)


    def forward(self, x, *args):
        out = self.linear(x)
        out = nn.Softmax(out)
        return out,x
    

        
class GatedSparseClassifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048,num_head=2, tau=1.0):
        super(GatedSparseClassifier, self).__init__()
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.head_class = num_classes//num_head
        self.linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_class, bias=False) for i in range(num_head)])
        self.tau = tau
        self.gateds = nn.ModuleList( Gated(self.head_class) for i in range(num_head-1))
        
    def forward(self, x, *args):
        output = []
        x_list = torch.split(x,self.head_dim,dim=1)
        gated = 0
        formal = 0
        for x,fc,gate in zip(x_list,self.linears,self.gateds):
            out = fc(x) + gated * formal
            gated,formal = gate(out)
            output.append(out)
        output = torch.cat(output, dim=1)
        output *= self.tau
        return output,None
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, tau=None, alpha=None, gamma=None, *args):
    print('Loading GatedSparseClassifier Classifier.')
    clf = GatedSparseClassifier(num_classes, feat_dim, num_head=num_head)
    return clf

if __name__ == '__main__':
    x = torch.randn((10,2048)).cuda()
    y = torch.randint(0,1000,(10,1)).squeeze(1).cuda()
    classifer = GatedSparseClassifier()
    target,_ = classifer(x)
    loss = nn.CrossEntropyLoss()(target,y)
    print(loss)

