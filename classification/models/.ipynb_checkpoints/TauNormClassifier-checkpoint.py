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
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import *
from os import path
import math

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048,num_head=2, **args):
        super(SparseClassifier, self).__init__()
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.head_class = num_classes//num_head
        self.linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_class, bias=False) for i in range(num_head)])
        self.scales = Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False
        

    def forward(self, x, label, embed):
        output = []
        x_list = torch.split(x,self.head_dim,dim=1)
        for x,fc in zip(x_list,self.linears):
            output.append(fc(x))
        x = torch.cat(output, dim=1)
        x *= self.scales
        return x, None
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading Tau Norm Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            if log_dir is not None:
                subdir = log_dir.strip('/').split('/')[-1]
                subdir = subdir.replace('stage2', 'stage1')
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading classifier weights from %s' % weight_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf