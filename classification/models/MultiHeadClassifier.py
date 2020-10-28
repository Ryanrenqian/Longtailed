"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""
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



class MultiHeadClassifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048,num_head=2, **args):
        super(MultiHeadClassifier, self).__init__()
        self.num_head = num_head
        self.head_dim = feat_dim // num_head 
        self.head_class = num_classes//num_head
        self.linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_class, bias=False) for i in range(num_head)])
        for fc in self.linears:
            self.reset_parameters(fc.weight)
    
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, *args):
        output = []
        x_list = torch.split(x,self.head_dim,dim=1)
        for x,fc in zip(x_list,self.linears):
            output.append(fc(x))
        output = torch.cat(output, dim=1)
        return output,None
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, tau=None, alpha=None, gamma=None, *args):
    print('Loading MultiHead Classifier.')
    clf = MultiHeadClassifier(num_classes, feat_dim, num_head=num_head)
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
