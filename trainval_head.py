# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
from pathlib import Path
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.autograd import gradcheck

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms

from model.faster_rcnn.vgg import vgg
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.densenet import densenet
from model.faster_rcnn.senet.se_resnet import se_resnet
from model.faster_rcnn.resnext import resnext
from model.faster_rcnn.mobilenet import mobilenet
from model.faster_rcnn.shufflenet import shufflenet
from model.faster_rcnn.squeezenet import squeezenet
from model.faster_rcnn.wide_resnet import wide_resnet

from model.faster_rcnn.faster_rcnn import _replace_module


def to_int_list(argument):
    f = lambda x: x.split(",")
    return list(map(int, f(argument)))
  
def to_str_list(argument):
    f = lambda x: x.split(",")
    return list(map(str, f(argument)))

def parse_args():
  """Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                     help='vgg16, res101',
                     default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=0, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  parser.add_argument('--imagenet_weight',
                      help='path to weight file of imagenet weight',
                      default=None, type=str)

  parser.add_argument('--save_epoch', dest='save_epoch',
                      help='per epoch to save models', default=1,
                      type=int)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=to_int_list)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# set val 
  parser.add_argument('--val', help='val per training epoch or not', 
                      action='store_true')

# set padding for conv layer
  parser.add_argument('--pad', help='padding mode for convolution layer', 
                      action='store_true')
  parser.add_argument('--pad_blocks', help='blocks to apply padding option',
                      default=None, type=to_str_list)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
  parser.add_argument('--tfb_dir', dest='tfb_dir', help='directory to save tensorboard result', default='tfb_log',
                      type=str)
  parser.add_argument('--head_train_types', dest='head_train_types', help='train_all, fixed_base, fixed_base_top', 
                      default=None)
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def val(epoch, fasterRCNN, cfg):
  print('=== start val in epoch {} ==='.format(epoch))

  # [val set]
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(args.imdbval_name, False)
  imdb_val.competition_mode(on=True)
  val_size = len(roidb_val)
  print('{:d} val roidb entries'.format(len(roidb_val)))
  cfg.TRAIN.USE_FLIPPED = True # change again for training

  # [val dataset]
  dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, 1, \
                               imdb_val.num_classes, training=False, normalize_as_imagenet=True)
  dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                               shuffle=False, num_workers=0)

  # print(' == forcibly insert checkpoint loading == ')
  # load_name = './models/ImgNet_pre/vgg16/coco/train_all/imagenet_0/head_1.pth'
  # print('load {}'.format(load_name))
  # checkpoint = torch.load(load_name)
  # fasterRCNN.load_state_dict(checkpoint['model'])

  output_dir = get_output_dir(imdb_val, 'val_in_training')
  data_iter_val = iter(dataloader_val)
  num_images = len(imdb_val.image_index)
  thresh = 0.0
  max_per_image = 100
  all_boxes = [[[] for _ in range(num_images)]
               for _ in range(imdb_val.num_classes)]

  # import ipdb; ipdb.set_trace()
  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  for i in range(num_images):
      data = next(data_iter_val)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      # rois_val, cls_prob_val, bbox_pred_val, \
      # rpn_loss_cls_val, rpn_loss_box_val, \
      # RCNN_loss_cls_val, RCNN_loss_bbox_val, \
      # rois_label_val = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      for j in range(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in range(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in range(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

  print('Evaluating detections')
  mAP = imdb_val.evaluate_detections(all_boxes, output_dir, result_file=None)
  del dataset_val, dataloader_val
  return mAP



if __name__ == '__main__':
  args = parse_args()
  print('Called with args:')
  print(args)

  if args.imagenet_weight:
    imagenet_weight_epoch = args.imagenet_weight.split('/')[-1].split('.')[0].split('_')[-1]
  else:
    imagenet_weight_epoch = 0

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_dist":
      args.imdb_name = "voc_2007_traindist"
      args.imdbval_name = "voc_2007_valdist"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # [train set]
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)
  print('{:d} train roidb entries'.format(len(roidb)))
  
  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + args.head_train_types + "/" + "imagenet_{}".format(imagenet_weight_epoch)
  # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + 'fixed-base_fixed-top'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # [for train]
  sampler_batch = sampler(train_size, args.batch_size)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, normalize_as_imagenet=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers, pin_memory=True)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  pretrained = True
  if args.net == 'vgg11':
    fasterRCNN = vgg(imdb.classes, 11, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'vgg13':
    fasterRCNN = vgg(imdb.classes, 13, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'vgg16':
    fasterRCNN = vgg(imdb.classes, 16, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'vgg19':
    fasterRCNN = vgg(imdb.classes, 19, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'resnet18':
    fasterRCNN = resnet(imdb.classes, 18, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'resnet34':
    fasterRCNN = resnet(imdb.classes, 34, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'resnet50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'resnet101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'resnet152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'densenet121':
    fasterRCNN = densenet(imdb.classes, 121, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'densenet161':
    fasterRCNN = densenet(imdb.classes, 161, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'densenet169':
    fasterRCNN = densenet(imdb.classes, 169, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)  
  elif args.net == 'densenet201':
    fasterRCNN = densenet(imdb.classes, 201, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  
  elif args.net == 'se_resnet50':
    fasterRCNN = se_resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'se_resnet101':
    fasterRCNN = se_resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'se_resnet152':
    fasterRCNN = se_resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'resnext50_32x4d':
    fasterRCNN = resnext(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'resnext101_32x8d':
    fasterRCNN = resnext(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'mobilenet_v2':
    fasterRCNN = mobilenet(imdb.classes, 'v2', pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'shufflenet_x05':
    fasterRCNN = shufflenet(imdb.classes, 'x05', pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'shufflenet_x10':
    fasterRCNN = shufflenet(imdb.classes, 'x10', pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'squeezenet_10':
    fasterRCNN = squeezenet(imdb.classes, '10', pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'squeezenet_11':
    fasterRCNN = squeezenet(imdb.classes, '11', pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  elif args.net == 'wide_resnet50':
    fasterRCNN = wide_resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)
  elif args.net == 'wide_resnet101':
    fasterRCNN = wide_resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic, imagenet_weight=args.imagenet_weight)

  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  if args.pad and len(args.pad_blocks)>0:
    # reflection padding
    if 'RCNN_base' in args.pad_blocks:
      _replace_module(fasterRCNN.RCNN_base)
    if 'RCNN_top' in args.pad_blocks:
      _replace_module(fasterRCNN.RCNN_top)
    if 'RCNN_rpn' in args.pad_blocks:
      _replace_module(fasterRCNN.RCNN_rpn)

  # import ipdb; ipdb.set_trace()

  # [check gradient]
  # # vgg16
  # fasterRCNN.RCNN_base[28].weight.register_hook(lambda grad: print('grad: last_conv\n{}'.format(grad[:3,:3,:3,:3])))
  # fasterRCNN.RCNN_cls_score.weight.register_hook(lambda grad: print('grad: linear_for_score \n{}'.format(grad[:, 10])))
  # # mbv2
  # fasterRCNN.RCNN_base[-1][0].weight.register_hook(lambda grad: print('grad: last_conv \n{}'.format(grad[:3,:3,:3,:3])))
  # fasterRCNN.RCNN_cls_score.weight.register_hook(lambda grad: print('grad: linear_for_score \n{}'.format(grad[:, 10])))
  # import ipdb; ipdb.set_trace()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  # fix base weight or not
  if args.head_train_types == 'fixed_base':
    print('=== fix base ===')
    for p in fasterRCNN.RCNN_base.parameters(): # layer1 - 3
        p.requires_grad_(False)
  elif args.head_train_types == 'fixed_base_top':
    print('=== fix base and top ===')
    for p in fasterRCNN.RCNN_base.parameters(): # layer1 - 3
        p.requires_grad_(False)
    for p in fasterRCNN.RCNN_top.parameters(): # layer4
        p.requires_grad_(False)
  elif args.head_train_types == 'train_all':
    print('=== train all parts ===')
    pass
  
  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    # load_name = os.path.join(output_dir,
    #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    load_name = args.resume
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    # tflogdir = "{}/{}/{}/{}".format(args.tfb_dir, args.net, args.head_train_types, imagenet_weight_epoch)
    tflogdir = os.path.join(args.tfb_dir, args.net, str(args.head_train_types), str(imagenet_weight_epoch))
    os.makedirs(tflogdir, exist_ok=True)
    logger = SummaryWriter(tflogdir)





  print("start: {} / max: {}".format(args.start_epoch, args.max_epochs))
  print("iteration per epoch : all images: {} / num batch: {}".format(len(dataset), len(dataloader)))

  if args.val:
    # [eval at initial weight]
    mAP = val(-1, fasterRCNN, cfg) # initial mAP
    if args.use_tfboard:
        info = {
          'val_mAP': mAP,
          }
        logger.add_scalars("logs_s_{}/val_mAP".format(args.session), info, 0)
    
  for epoch in range(args.start_epoch+1, args.max_epochs+1):
      # setting to train mode
      # import ipdb; ipdb.set_trace()
      fasterRCNN.train()
      
      loss_temp = 0
      start = time.time()

      # print('{} % {} = {}'.format(epoch+1, args.lr_decay_step+1, (epoch+1)%(args.lr_decay_step+1)))
      # if (epoch) % (args.lr_decay_step) == 0:
      if epoch in args.lr_decay_step:
          adjust_learning_rate(optimizer, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
        fasterRCNN.zero_grad()
        
        # import ipdb; ipdb.set_trace()
        # compute
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.net.startswith("vgg"):
        # if args.net.startswith('vgg') or args.net.startswith('shuffle') or args.net.startswith('squeeze'):
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          else:
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

          print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                        
          if args.use_tfboard:
            info = {
              'loss': loss_temp,
              'loss_rpn_cls': loss_rpn_cls,
              'loss_rpn_box': loss_rpn_box,
              'loss_rcnn_cls': loss_rcnn_cls,
              'loss_rcnn_box': loss_rcnn_box
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()


      if (epoch) == 1 or (epoch)%args.save_epoch == 0 or (epoch) == args.max_epochs:
        save_name = os.path.join(output_dir, 'head_{}.pth'.format(epoch))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch,
          'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
      
      if args.val:
        # [val step]
        mAP = val(epoch, fasterRCNN, cfg)
        # mAP = val_v2(epoch, cfg, save_name)

        print('outside mPA: {}'.format(mAP))
        if args.use_tfboard:
            info = {
              'val_mAP': mAP,
              }
            logger.add_scalars("logs_s_{}/val_mAP".format(args.session), info, (epoch) * iters_per_epoch)

  if args.use_tfboard:
    logger.close()
