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

# __all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

# model_urls = {
#   'densenet121': 'https://s3.amazonaws.com/pytorch/models/densenet121-a639ec97.pth',
#   'densenet161': 'https://s3.amazonaws.com/pytorch/models/densenet161-8d451a50.pth',
#   'densenet169': 'https://s3.amazonaws.com/pytorch/models/densenet169-b2777c0a.pth',
#   'densenet201': 'https://s3.amazonaws.com/pytorch/models/densenet201-c1103571.pth',
# }
'''difference in state_dict keys
model_zoo:          features.denseblock4.denselayer14.norm.2.weight
torchvision.models: features.denseblock4.denselayer14.norm2.weight
'''

dout_base_model = {
  121: 1024, # 
  169: 1280, # 
  201: 1792, # 
}

dout_top = {
  121: 1024, #
  169: 1664, # 
  201: 1920, # 

}

def densenet121(pretrained=False, imagenet_weight=False):
  """Constructs a DenseNet-121 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.densenet121()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      # model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
      model = models.densenet121(pretrained=True)
  return model


def densenet161(pretrained=False, imagenet_weight=False):
  """Constructs a DenseNet-161 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.densenet161()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      # model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
      model = models.densenet161(pretrained=True)
  return model


def densenet169(pretrained=False, imagenet_weight=False):
  """Constructs a DenseNet-169 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.densenet169()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      # model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
      model = models.densenet169(pretrained=True)
  return model


def densenet201(pretrained=False, imagenet_weight=False):
  """Constructs a DenseNet-201 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.densenet201()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      # model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
      model = models.densenet201(pretrained=True)
  return model


class densenet(_fasterRCNN):
  def __init__(self, classes, num_layers=169, pretrained=False, class_agnostic=False, imagenet_weight=None):
    self.dout_base_model = dout_base_model[num_layers] # depth of RCNN_base output. pattern1=1280 / pattern2=640 (in dense169) 
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.num_layers = num_layers
    self.imagenet_weight = imagenet_weight
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers == 121:
      densenet = densenet121(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 161:
      densenet = densenet161(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 169:
      densenet = densenet169(self.pretrained, self.imagenet_weight)
    elif self.num_layers == 201:
      densenet = densenet201(self.pretrained, self.imagenet_weight)
    else:
      raise ValueError('layers should be in [121, 161, 169, 201].')
	  
    # if self.pretrained == True:
    #   print("Loading pretrained weights from %s" %(self.model_path))
    #   state_dict = torch.load(self.model_path)
    #   densenet.load_state_dict({k:v for k,v in state_dict.items() if k in densenet.state_dict()})

    # Build densenet: === 
    # # pattern1 : back = ~denseblock3 / top = transition3~
    # self.RCNN_base = nn.Sequential(
    #   densenet.features.conv0, densenet.features.norm0, densenet.features.relu0, densenet.features.pool0, 
    #   densenet.features.denseblock1, densenet.features.transition1,
    #   densenet.features.denseblock2, densenet.features.transition2,
    #   densenet.features.denseblock3, # [1,1280,16,16]
    # )
    # self.RCNN_top = nn.Sequential(      
    #   densenet.features.transition3,
    #   densenet.features.denseblock4,
    #   densenet.features.norm5, # [1,1664,8,8]
    # )

    # # pattern2 : back = ~transition3 / top = denseblock4~
    # self.RCNN_base = nn.Sequential(
    #   densenet.features.conv0, densenet.features.norm0, densenet.features.relu0, densenet.features.pool0, 
    #   densenet.features.denseblock1, densenet.features.transition1,
    #   densenet.features.denseblock2, densenet.features.transition2,
    #   densenet.features.denseblock3, densenet.features.transition3, # [1,640,8,8]
    # )
    # self.RCNN_top = nn.Sequential(
    #   densenet.features.denseblock4,
    #   densenet.features.norm5, # [1,1664,8,8]
    # )


    # pattern3 : back = ~transition3 / top = denseblock4~
    self.RCNN_base = densenet.features[:9]
    self.RCNN_top = densenet.features[9:]
    # ===

    self.RCNN_cls_score = nn.Linear(dout_top[self.num_layers], self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(dout_top[self.num_layers], 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(dout_top[self.num_layers], 4*self.n_classes)

    # # === fix weight
    # # Fix blocks
    # for p in self.RCNN_base[0].parameters(): p.requires_grad=False # conv0
    # for p in self.RCNN_base[1].parameters(): p.requires_grad=False # norm0 ###

    # assert (0 <= cfg.DENSENET.FIXED_BLOCKS < 7)
    # if cfg.DENSENET.FIXED_BLOCKS >= 6:
    #   for p in self.RCNN_base[9].parameters(): p.requires_grad=False
    # if cfg.DENSENET.FIXED_BLOCKS >= 5:
    #   for p in self.RCNN_base[8].parameters(): p.requires_grad=False
    # if cfg.DENSENET.FIXED_BLOCKS >= 4:
    #   for p in self.RCNN_base[7].parameters(): p.requires_grad=False
    # if cfg.DENSENET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    # if cfg.DENSENET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    # if cfg.DENSENET.FIXED_BLOCKS >= 1:
    #   for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    # === 
    
    # def set_bn_fix(m):
    #   classname = m.__class__.__name__
    #   if classname.find('BatchNorm') != -1:
    #     for p in m.parameters(): p.requires_grad=False

    # self.RCNN_base.apply(set_bn_fix)
    # self.RCNN_top.apply(set_bn_fix) ###

  # def train(self, mode=True):
  #   # Override train so that the training mode is set as we want
  #   nn.Module.train(self, mode)
  #   if mode:
  #     # Set fixed blocks to be in eval mode
  #     self.RCNN_base.eval()
  #     print('train {} to {}'.format(cfg.DENSENET.FIXED_BLOCKS+4, len(self.RCNN_base)))
  #     for i in range(cfg.DENSENET.FIXED_BLOCKS+4, len(self.RCNN_base)): # 1->5-, 2->6- 
  #         self.RCNN_base[i].train()
        
  #     def set_bn_eval(m):
  #       classname = m.__class__.__name__
  #       if classname.find('BatchNorm') != -1:
  #         m.eval()

  #     self.RCNN_base.apply(set_bn_eval)
  #     self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5) # [1,1664]
    fc7 = F.relu(fc7, inplace=True) # relu 
    fc7 = F.adaptive_avg_pool2d(fc7, (1,1)) # avgpool
    fc7 = torch.flatten(fc7, 1)
    return fc7

# function to load weight
def exchange_weightkey_in_state_dict(state_dict):
    new_state_dict= OrderedDict()
    for k, v in state_dict.items():
        name=k[7:] #remove 'module.' of DataParallel
        new_state_dict[name] = v
    return new_state_dict
