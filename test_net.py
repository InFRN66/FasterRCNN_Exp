# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
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

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg import vgg
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.densenet import densenet
from model.faster_rcnn.senet.se_resnet import se_resnet
from model.faster_rcnn.resnext import resnext
from model.faster_rcnn.mobilenet import mobilenet
from model.faster_rcnn.shufflenet import shufflenet
from model.faster_rcnn.squeezenet import squeezenet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--trained_model', dest='trained_model',
                      help='traied model to restore images',
                      default=None, type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--result_file', dest='result_file',
                      help='file path to save result',
                      default=None, type=str)
  parser.add_argument('--load_name', dest='load_name',
                      help='file path to load model',
                      default=None, type=str)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  # if not os.path.isdir(Path(args.result_file).parent):
  #   os.makedirs(Path(args.result_file).parent, exist_ok=True)
    
  # print('Called with args:')
  # print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  imagenet_weight_epoch = args.load_name.split('/')[-2].split('_')[-1]
  print('imagenet: {}'.format(imagenet_weight_epoch))

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_dist":
      args.imdb_name = "voc_2007_traindist"
      args.imdbval_name = "voc_2007_valdist"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  
  # add another dataset
  if args.dataset in [
            'brightness', 'defocus_blur', 'fog', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'shot_noise', 'zoom_blur', 'contrast', 'elastic_transform', 'frost', 'glass_blur', 'jpeg_compression', 'pixelate', 'snow'
        ]:
      args.imdb_name = "{}_2007_traindist".format(args.dataset)
      args.imdbval_name = "{}_2007_valdist".format(args.dataset)
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  if args.dataset in [
            'brightness', 'defocus_blur', 'fog', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'shot_noise', 'zoom_blur', 'contrast', 'elastic_transform', 'frost', 'glass_blur', 'jpeg_compression', 'pixelate', 'snow'
        ] and args.trained_model:
      args.imdb_name = "{}_trained_{}_2007_traindist".format(args.trained_model, args.dataset)
      args.imdbval_name = "{}_trained_{}_2007_valdist".format(args.trained_model, args.dataset)
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)
  print('{:d} roidb entries'.format(len(roidb)))

  # if args.dataset == "pascal_voc_dist":
  #   input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  # else:
  #   input_dir = args.load_dir + "/" + args.net + "/" + "pascal_voc_dist+" + args.dataset
  
  # if not os.path.exists(input_dir):
  #   raise Exception('There is no input directory for loading network from ' + input_dir)
  # # load_name = os.path.join(input_dir,
  # #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  
  load_name = args.load_name


  # initilize the network here.
  '''('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
  '''
  pretrained = False
  if args.net == 'vgg11':
    fasterRCNN = vgg(imdb.classes, 11, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg13':
    fasterRCNN = vgg(imdb.classes, 13, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg16':
    fasterRCNN = vgg(imdb.classes, 16, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg19':
    fasterRCNN = vgg(imdb.classes, 19, pretrained=pretrained, class_agnostic=args.class_agnostic)

  elif args.net == 'resnet18':
    fasterRCNN = resnet(imdb.classes, 18, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'resnet34':
    fasterRCNN = resnet(imdb.classes, 34, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'resnet50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'resnet101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'resnet152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=args.class_agnostic)

  elif args.net == 'densenet121':
    fasterRCNN = densenet(imdb.classes, 121, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'densenet161':
    fasterRCNN = densenet(imdb.classes, 161, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'densenet169':
    fasterRCNN = densenet(imdb.classes, 169, pretrained=pretrained, class_agnostic=args.class_agnostic)  
  elif args.net == 'densenet201':
    fasterRCNN = densenet(imdb.classes, 201, pretrained=pretrained, class_agnostic=args.class_agnostic)
  
  elif args.net == 'se_resnet50':
    fasterRCNN = se_resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'se_resnet101':
    fasterRCNN = se_resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'se_resnet152':
    fasterRCNN = se_resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=args.class_agnostic)\

  elif args.net == 'resnext50':
    fasterRCNN = resnext(imdb.classes, 50, pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'resnext101':
    fasterRCNN = resnext(imdb.classes, 101, pretrained=pretrained, class_agnostic=args.class_agnostic)
  
  elif args.net == 'mobilenet_v2':
    fasterRCNN = mobilenet(imdb.classes, 'v2', pretrained=pretrained, class_agnostic=args.class_agnostic)

  elif args.net == 'shufflenet_x05':
    fasterRCNN = shufflenet(imdb.classes, 'x05', pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'shufflenet_x10':
    fasterRCNN = shufflenet(imdb.classes, 'x10', pretrained=pretrained, class_agnostic=args.class_agnostic)

  elif args.net == 'squeezenet_10':
    fasterRCNN = squeezenet(imdb.classes, '10', pretrained=pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'squeezenet_11':
    fasterRCNN = squeezenet(imdb.classes, '11', pretrained=pretrained, class_agnostic=args.class_agnostic)

  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')

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

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in range(num_images)]
               for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                           imdb.num_classes, training=False, normalize=False, normalize_as_imagenet=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

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
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
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
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
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

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  mAP = imdb.evaluate_detections(all_boxes, output_dir, args.result_file)
  print('outside mPA: {}'.format(mAP))

  end = time.time()
  print("test time: %0.4fs" % (end - start))
