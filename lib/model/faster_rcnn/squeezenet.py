# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

dout_base_model = {
  '10': 512,
  '11': 512,
}

dout_top = {
  '10': 512,
  '11': 512,
}

def squeezenet_10(pretrained=False, imagenet_weight=False):
    """Constructs a squeeze-Net 10 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.squeezenet1_0()
    if pretrained:
        if imagenet_weight:
            print('=== use {} as backbone'.format(imagenet_weight))
            state_dict = torch.load(imagenet_weight)['state_dict']
            state_dict = exchange_weightkey_in_state_dict(state_dict)
            model.load_state_dict(state_dict)
        else:
            print('=== use pytorch default backbone')
            mode1l = models.squeezenet1_0(pretrained=True)
    return model


def squeezenet_11(pretrained=False, imagenet_weight=False):
    """Constructs a squeeze-Net 11 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.squeezenet1_1()
    if pretrained:
        if imagenet_weight:
            print('=== use {} as backbone'.format(imagenet_weight))
            state_dict = torch.load(imagenet_weight)['state_dict']
            state_dict = exchange_weightkey_in_state_dict(state_dict)
            model.load_state_dict(state_dict)
        else:
            print('=== use pytorch default backbone')
            model = models.squeezenet1_1(pretrained=True)
    return model


class squeezenet(_fasterRCNN):
    def __init__(self, classes, ver='10', pretrained=False, class_agnostic=False, imagenet_weight=None):
        # dim of output from RCNN_base block
        self.dout_base_model = dout_base_model[ver]
        self.ver = ver
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.imagenet_weight = imagenet_weight

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        if self.ver == '10':
            squeezenet = squeezenet_10(self.pretrained, self.imagenet_weight)
        elif self.ver == '11':
            squeezenet = squeezenet_11(self.pretrained, self.imagenet_weight)
        else:
            raise ValueError('version should be in [10, 11].')

        # if self.pretrained:
        #     print("Loading pretrained weights from %s" %(self.model_path))
        #     state_dict = torch.load(self.model_path)
        #     vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

        # [drop, conv, relu, pool] -> [drop]
        # squeezenet.classifier = nn.Sequential(
        #     list(squeezenet.classifier._modules.values())[0]
        # )

        # not using the last maxpool layer
        # self.RCNN_base = nn.Sequential(
        #     *list(squeezenet.features._modules.values())
        # )
        self.RCNN_base = nn.Sequential(
            *list(squeezenet.features._modules.values())[:12]
        )
      
        # # Fix the layers [conv, relu, pool]:
        # for layer in range(3):
        #     for p in self.RCNN_base[layer].parameters():
        #         p.requires_grad = False

        # self.RCNN_top = squeezenet.classifier # apply just classifier part
        # self.RCNN_top = nn.Sequential() # pass through
        self.RCNN_top = nn.Sequential(
            *list(squeezenet.features._modules.values())[12:]
        )

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(dout_top[self.ver], self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(dout_top[self.ver], 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(dout_top[self.ver], 4*self.n_classes)

    def _head_to_tail(self, pool5):
        # print('head_to_tail: {}'.format(pool5.shape))
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        # print('fc7: {}'.format(fc7.shape))
        return fc7

# function to load weight
def exchange_weightkey_in_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.' of DataParallel
        name = k.replace('features.module', 'features')
        new_state_dict[name] = v
    return new_state_dict
