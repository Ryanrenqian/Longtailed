# from utils import *
from os import path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BBN_ResNet_Cifar(nn.Module):
    def __init__(self, block, planes=[16,32,64],strides=[1,1,2],num_blocks=[5,5,5], depth=0, branch=1):
        super(BBN_ResNet_Cifar, self).__init__()
        in_planes = planes[0]
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.share_layers, in_planes = self.generate_share(block, in_planes,planes, strides, num_blocks, depth)
        self.branchs = nn.ModuleList([ self.generate_branch(block, in_planes,planes, strides, num_blocks, depth) for i in range(branch)])
#         print(self.branchs[0])
        self.apply(_weights_init)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def generate_branch(self, block, in_planes, planes,strides, num_blocks, depth):
        branch_layers = []
        set_branch = False
        all_blocks = sum(num_blocks)
        for i,num_block in enumerate(num_blocks):
            if all_blocks - depth<num_block and not set_branch:
                set_branch = True
                branch_layers.append(self._make_layer(block,in_planes,planes[i],depth+num_block-all_blocks,stride=1))
                in_planes = planes[i]*block.expansion
            elif set_branch:
                branch_layers.append(self._make_layer(block,in_planes,planes[i],num_block,stride=strides[i]))
                in_planes = planes[i]*block.expansion
            all_blocks -= num_block
        return nn.Sequential(*branch_layers)
    
    def generate_share(self, block, in_planes,planes,strides, num_blocks, depth):
        share_layers = []
        all_blocks = sum(num_blocks)
        for i,num_block in enumerate(num_blocks):
            if all_blocks - depth<num_block:
                share_layers.append(self._make_layer(block,in_planes,planes[i],all_blocks-depth,stride=strides[i]))
                break
            else:
                share_layers.append(self._make_layer(block,in_planes,planes[i],num_block,stride=strides[i]))
            in_planes=planes[i]*block.expansion
            all_blocks -= num_block

        return nn.Sequential(*share_layers),planes[i]


    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)["state_dict_best"]['feat_model']

        new_dict = OrderedDict()

        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self,block, in_planes, planes, num_blocks, stride, add_flag=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
#             print('block',in_planes, planes,stride)
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.share_layers(out)
#         print(out.size())
        if len(self.branchs)>0:
#             print(self.branchs[0](out))
            out = torch.cat([branch(out) for branch in self.branchs], dim=1)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
        # if "feature_cb" in kwargs:
        #     out = self.cb_block(out)
        #     return out
        # elif "feature_rb" in kwargs:
        #     out = self.rb_block(out)
        #     return out

        # out1 = self.cb_block(out)
        # out2 = self.rb_block(out)
        # out = torch.cat((out1, out2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)


        return out


        
def create_model(use_fc=False, pretrain=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, depth=0,branch=0):
    
    print('Loading ResNet 32 Feature Model.')
    resnet32 = BBN_ResNet_Cifar(BasicBlock,planes=[16,32,64],strides=[1,2,2],num_blocks=[5,5,5],depth=depth,branch=branch)

    pretrained_model="./data/checkpoints/final_model_checkpoint.pth"
    if path.exists(pretrained_model) and pretrain:
        print('===> Load Initialization for ResNet32')
        resnet32.load_model(pretrain=pretrained_model)
    else:
        print('===> Train backbone from the scratch')

    return resnet32

if __name__ =='__main__':
    model =create_model(depth=8,branch=4).cuda()
    from torchsummary import summary
    summary(model.cuda(), (3, 32, 32))
#     from tensorboardX import SummaryWriter
#     import numpy as np
#     with SummaryWriter() as w:
#         w.add_graph(model,np.zeros((32,32)))
#     checkpoint = '/root/workspace/imbalanced-semi-self/resnet32/cifar100_resnet32_CE_None_exp_0.01_ss_pretrained/ckpt.best.pth.tar'
#     data = torch.load(checkpoint,'cpu')
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in data['state_dict'].items():
#         if  'fc' not in k:
#             new_state_dict[k] = v
    
#     model.load_state_dict(new_state_dict)
#     new_state_dict = model.state_dict()
#     new_state_dict['classifier'] = data['state_dict']['fc.weight']
#     best_model_weights = {}
#     best_model_weights['feat_model'] = model.state_dict()
#     best_model_weights['classifier'] = {'classifier':data['state_dict']['fc.weight']}
#     model_states={'state_dict': best_model_weights}
#     torch.save(model_states,'/root/workspace/Long-Tailed-Recognition.pytorch/classification/logs/CIFAR100_LT/models/resnet32s_ssp_e200_ratio100/latest_model_checkpoint.pth')