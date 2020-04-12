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
import pdb

from torchvision import models

__all__ = ['resnext50_32x4d', 'resnext101_32x8d']

dout_base_model = {
  50: 1024,
  101: 1024,
}

def resnext50(pretrained=False, imagenet_weight=False):
  """Constructs a ResNext-50_32x4d model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  print('=== Using resnet 50_32x4d ===')
  model = models.resnext50_32x4d()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model = models.resnext50_32x4d(pretrained=True)
  return model


def resnext101(pretrained=False, imagenet_weight=False):
  """Constructs a ResNext-101_32x8d model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  print('=== Using resnet 101_32x8d ===')
  model = models.resnext101_32x8d()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model = models.resnext101_32x8d(pretrained=True)
  return model


class resnext(_fasterRCNN):
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
      resnext = resnext50(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 101:
      resnext = resnext101(self.pretrained, self.imagenet_weight)
    else:
      raise ValueError('layers should be in [50, 101].')

    # if self.pretrained == True:
    #   print("Loading pretrained weights from %s" %(self.model_path))
    #   state_dict = torch.load(self.model_path)
    #   resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnext.
    self.RCNN_base = nn.Sequential(resnext.conv1, resnext.bn1, resnext.relu,
      resnext.maxpool, resnext.layer1, resnext.layer2, resnext.layer3) # until layer3

    self.RCNN_top = nn.Sequential(resnext.layer4) # layer4

    self.RCNN_cls_score = nn.Linear(dout_base_model[self.num_layers]*2, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(dout_base_model[self.num_layers]*2, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(dout_base_model[self.num_layers]*2, 4*self.n_classes)

    # === fix weight
    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    print('cfg.RESNET.FIXED_BLOCKS: {}'.format(cfg.RESNET.FIXED_BLOCKS))
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    # ==== === ====

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1: # if batchnorm
        for p in m.parameters(): p.requires_grad=False

    # === fix weight
    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      print('train {}to{}'.format(cfg.RESNET.FIXED_BLOCKS+4, len(self.RCNN_base)))
      for i in range(cfg.DENSENET.FIXED_BLOCKS+4, len(self.RCNN_base)): # 1->5-, 2->6- 
          self.RCNN_base[i].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

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