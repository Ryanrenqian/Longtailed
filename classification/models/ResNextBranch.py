"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import math
import torch.nn as nn
import torch.nn.functional as F
import torch
# from utils import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, is_last=False, last_relu=True):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.is_last and (not self.last_relu):
            pass
        else:
            out = self.relu(out)

        return out

class ResNext(nn.Module):

    def __init__(self, block, layers, branch=1, depth=4, planes=[64,128,256,512], strides=[1,2,2,2], groups=1, width_per_group=64, use_fc=False, dropout=None,use_glore=False, use_gem=False, last_relu=True):
#         self.inplanes = 64
        super(ResNext, self).__init__()
        self.branch = branch
        self.depth = depth
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.share_layers,inplanes = self.generate_share(block, 64, planes, strides, layers, depth)
        self.branch_layers = nn.ModuleList([ self.generate_branch(block, inplanes, planes, strides, layers, branch, depth) for i in range(branch)])
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.use_fc = use_fc
        self.use_dropout = True if dropout else False
    
    def generate_branch(self, block, in_planes, planes,strides, layers, branch, depth):
        branch_layers = []
        for i in range(depth,len(layers)):
            if i != 3:
                layer,in_planes = self._make_layer(block,in_planes, planes[i]//branch, layers[i], strides[i])
            else:
                layer,in_planes = self._make_layer(block,in_planes, planes[i]//branch, layers[i], strides[i], is_last=False, last_relu=True)
            branch_layers.append(layer)
        return nn.Sequential(*branch_layers)
        
    def generate_share(self, block, in_planes, planes, strides, layers, depth, is_last=False, last_relu=True):
        share_layers = []
        for i in range(depth):
            if i != 3:
                layer,in_planes = self._make_layer(block, in_planes, planes[i],layers[i], strides[i])
            else:
                layer,in_planes = self._make_layer(block, in_planes, planes[i],layers[i], strides[i], is_last=False, last_relu=True)
            share_layers.append(layer)
        return nn.ModuleList(share_layers),in_planes
#         return nn.Sequential(*share_layers),in_planes
        

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, is_last=False, last_relu=True):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks-1), last_relu=last_relu))
        print('block',inplanes, planes,blocks,stride)
        return nn.Sequential(*layers),inplanes

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.share_layers:
            x = layer(x)
#         x = self.share_layers(x)
#         x_list = torch.split(x,self.branch,dim=1)
        out = []
        for branch in self.branch_layers:
            out.append(branch(x))
        x = torch.cat(out,dim=1)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        if self.use_fc:
            x = F.relu(self.fc_add(x))

        if self.use_dropout:
            x = self.dropout(x)

        return x
    
if __name__ == '__main__':
    model = ResNext(Bottleneck, layers=[3, 4, 6, 3],branch=2,depth=3, groups=32, width_per_group=4)
    from torchsummary import summary
    summary(model.cuda(), (3, 256, 256))