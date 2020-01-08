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
  'v1': None,
  'v2': 1280,
  'v3': None,
}


def mobilenet_v2(pretrained=False, imagenet_weight=False):
  """Constructs a MobileNet-v2 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = models.mobilenet_v2()
  if pretrained:
    if imagenet_weight:
      print('=== use {} as backbone'.format(imagenet_weight))
      state_dict = torch.load(imagenet_weight)['state_dict']
      state_dict = exchange_weightkey_in_state_dict(state_dict)
      model.load_state_dict(state_dict)
    else:
      print('=== use pytorch default backbone')
      model = models.mobilenet_v2(pretrained=True)
  return model


class mobilenet(_fasterRCNN):
  def __init__(self, classes, version='v2', pretrained=False, class_agnostic=False, imagenet_weight=None):
    self.dout_base_model = dout_base_model[version] # dim of output from RCNN_base block
    self.version = version
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.imagenet_weight = imagenet_weight
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.version == 'v1':
      mbnet = None
    elif self.version == 'v2':
      mbnet = mobilenet_v2(self.pretrained, self.imagenet_weight)
    elif self.version == 'v3':
      mbnet = None
    else:
      raise ValueError('version should be in [v1, v2, v3].')
    
    # if self.pretrained:
    #     print("Loading pretrained weights from %s" %(self.model_path))
    #     state_dict = torch.load(self.model_path)
    #     mbnet.load_state_dict({k:v for k,v in state_dict.items() if k in mbnet.state_dict()})

    mbnet.classifier = nn.Sequential(*list(mbnet.classifier._modules.values())[:-1]) # only Dropout

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(mbnet.features._modules.values()))

    # Fix some conv layers:
    for layer in range(3):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(mbnet.features, self.classes, self.dout_base_model)

    self.RCNN_top = mbnet.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(self.dout_base_model, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(self.dout_base_model, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(self.dout_base_model, 4*self.n_classes)      

  def _head_to_tail(self, pool5): # [b*256, 1280, 7, 7] -> [b*256, 1280]
    # print('head_to_tail: {}'.format(pool5.shape))
    pool5_flat = pool5.mean(3).mean(2)
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
