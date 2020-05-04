from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from torchvision import models
import pdb

__all__ = ['WideResNet', 'wide_resnet50', 'wide_resnet101']


model_urls = {
  'wide_resnet50': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
  'wide_resnet101': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

dout_base_model = {
  50: 1024,
  101: 1024,
}

dout_top = {
  50:  2048,
  101: 2048,
}

def wide_resnet50(pretrained=False, imagenet_weight=False):
  """Constructs a Wide_ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  print('=== Using wide_resnet 50 ===')
  model = models.wide_resnet50_2()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model = models.wide_resnet50_2(pretrained=True)
  return model


def wide_resnet101(pretrained=False, imagenet_weight=False):
  """Constructs a Wide_ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  print('=== Using wide_resnet 101 ===')
  model = models.wide_resnet101_2()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model = models.wide_resnet101_2(pretrained=True)
  return model


class wide_resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, imagenet_weight=None):
    # self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = dout_base_model[num_layers]
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.num_layers = num_layers
    self.imagenet_weight = imagenet_weight

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers == 50:
      wide_resnet = wide_resnet50(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 101:
      wide_resnet = wide_resnet101(self.pretrained, self.imagenet_weight)
    else:
      raise ValueError('layers should be in [50, 101].')

    # if self.pretrained == True:
    #   print("Loading pretrained weights from %s" %(self.model_path))
    #   state_dict = torch.load(self.model_path)
    #   wide_resnet.load_state_dict({k:v for k,v in state_dict.items() if k in wide_resnet.state_dict()})

    # Build wide_resnet.
    self.RCNN_base = nn.Sequential(wide_resnet.conv1, wide_resnet.bn1, wide_resnet.relu,
      wide_resnet.maxpool, wide_resnet.layer1, wide_resnet.layer2, wide_resnet.layer3) # until layer3

    self.RCNN_top = nn.Sequential(wide_resnet.layer4) # layer4

    self.RCNN_cls_score = nn.Linear(dout_top[self.num_layers], self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(dout_top[self.num_layers], 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(dout_top[self.num_layers], 4*self.n_classes)

    # # === fix weight
    # # Fix blocks
    # for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    # for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # print('cfg.RESNET.FIXED_BLOCKS: {}'.format(cfg.RESNET.FIXED_BLOCKS))
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 1:
    #   for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    # # ==== === ====

    # def set_bn_fix(m):
    #   classname = m.__class__.__name__
    #   if classname.find('BatchNorm') != -1:
    #     for p in m.parameters(): p.requires_grad=False

    # self.RCNN_base.apply(set_bn_fix)
    # self.RCNN_top.apply(set_bn_fix)

  # def train(self, mode=True):
  #   # Override train so that the training mode is set as we want
  #   nn.Module.train(self, mode)
  #   if mode:
  #     # Set fixed blocks to be in eval mode
  #     self.RCNN_base.eval()
  #     print('train {}to{}'.format(cfg.RESNET.FIXED_BLOCKS+4, len(self.RCNN_base)))
  #     for i in range(cfg.DENSENET.FIXED_BLOCKS+4, len(self.RCNN_base)): # 1->5-, 2->6- 
  #         self.RCNN_base[i].train()

  #     def set_bn_eval(m):
  #       classname = m.__class__.__name__
  #       if classname.find('BatchNorm') != -1:
  #         m.eval()

  #     self.RCNN_base.apply(set_bn_eval)
  #     self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    # print('head_to_tail: {}'.format(pool5.shape))
    fc7 = self.RCNN_top(pool5).mean(3).mean(2) # pool5(feature)にRCNN_TOPかけてmean(3).mean(2)
    # print('fc7: {}'.format(fc7.shape))
    return fc7


# function to load weight
def exchange_weightkey_in_state_dict(state_dict):
    new_state_dict= OrderedDict()
    for k, v in state_dict.items():
        name=k[7:] #remove 'module.' of DataParallel
        new_state_dict[name] = v
    return new_state_dict