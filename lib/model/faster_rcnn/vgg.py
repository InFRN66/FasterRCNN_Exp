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

model_urls = {
  'vgg11': 'https://s3.amazonaws.com/pytorch/models/vgg11-bbd30ac9.pth',
  'vgg13': 'https://s3.amazonaws.com/pytorch/models/vgg13-c768596a.pth',
  'vgg16': 'https://s3.amazonaws.com/pytorch/models/vgg16-397923af.pth',
  'vgg19': 'https://s3.amazonaws.com/pytorch/models/vgg19-dcbb9e9d.pth',
}

dout_base_model = {
  11: 512,
  13: 512,
  16: 512,
  19: 512
}


def vgg11(pretrained=False, imagenet_weight=False):
  """Constructs a VGG-11 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.vgg11()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
  return model


def vgg13(pretrained=False, imagenet_weight=False):
  """Constructs a VGG-13 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.vgg13()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
  return model


def vgg16(pretrained=False, imagenet_weight=False):
  """Constructs a VGG-16 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.vgg16()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
  return model


def vgg19(pretrained=False, imagenet_weight=False):
  """Constructs a VGG-19 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.vgg19()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
  return model


class vgg(_fasterRCNN):
  def __init__(self, classes, num_layers=16, pretrained=False, class_agnostic=False, imagenet_weight=None):
    # self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = dout_base_model[num_layers] # dim of output from RCNN_base block
    self.num_layers = num_layers
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.imagenet_weight = imagenet_weight

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers == 11:
      vgg = vgg11(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 13:
      vgg = vgg13(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 16:
      vgg = vgg16(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 19:
      vgg = vgg19(self.pretrained, self.imagenet_weight)
    else:
      raise ValueError('layers should be in [11, 13, 16, 19].')
    
    # if self.pretrained:
    #     print("Loading pretrained weights from %s" %(self.model_path))
    #     state_dict = torch.load(self.model_path)
    #     vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # # === fix weight 
    # # Fix the layers before conv3:
    # for layer in range(10):
    #   for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
    # # === 

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4*self.n_classes)      

  def _head_to_tail(self, pool5): # [b*256, 612 7, 7] -> [b*256, 4096]
    # print('head_to_tail: {}'.format(pool5.shape))
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    # print('fc7: {}'.format(fc7.shape))
    return fc7

# function to load weight
def exchange_weightkey_in_state_dict(state_dict):
    new_state_dict= OrderedDict()
    for k, v in state_dict.items():
        name=k.replace('features.module', 'features') #remove 'module.' of DataParallel
        new_state_dict[name] = v
    return new_state_dict
