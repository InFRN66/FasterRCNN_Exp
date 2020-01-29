import torch
import numpy as np
import os, sys
from collections import namedtuple, defaultdict
import glob
import matplotlib.pyplot as plt


def data_extraction_clf(net, path_prefix='../../examples/imagenet/val'):
    path = os.path.join(path_prefix, net+'_val')
    if not os.path.isfile(path):
        return -1
    with open(path, 'r') as f:
        data = f.readline().split(',')
        score = namedtuple('result', ['acc1', 'acc5'])
        score.acc1 = float(data[0].strip())*0.01
        score.acc5 = float(data[1].strip())*0.01
    return score


def data_extraction_det(net, path_prefix='output/ImgNet_pre/train_all'):
    path = os.path.join(path_prefix, net+'_det.txt')
    if not os.path.isfile(path):
        return -1
    result = []
    with open(path, 'r') as f:
        for i, data in enumerate(f.readlines()):
            score = namedtuple('result', ['range', 'AP'])
            score.range = data[28:41].strip()
            score.AP = float(data.split('=')[-1].strip())
            result.append(score)
            if i >= 2:
                break
    return result


def data_plot(result, pdir_det, clf_score='acc1'):
    def label_def(net):
        if net.startswith('vgg'):
            return 'o'
        elif net.startswith('resnet'):
            return '^'
        elif net.startswith('densenet'):
            return 's'
        elif net.startswith('se_'):
            return '*'
        elif net.startswith('resnext'):
            return 'd'
        elif net.startswith('shuffle'):
            return '+'
        elif net.startswith('mobile'):
            return '_'
       
    plt.figure(figsize=(10,6))
    for net in result.keys():
        data = result[net]
        CLF = data.clf
        DET = data.det
        x = {'acc1': CLF.acc1, 'acc5': CLF.acc5}        
        y = DET[0].AP # IOU=0.5:0.95
        plt.scatter(x[clf_score], y, label=net, marker=label_def(net), s=100, alpha=0.5)
    plt.legend()
    plt.xlim(0.5, 1)
    plt.ylim(0, 0.5)
    plt.xlabel('{}'.format(clf_score))
    plt.ylabel('AP ({})'.format(DET[0].range))
    plt.grid()
    if pdir_det.find('train_all'):
        plt.title('train_all')
    elif pdir_det.find('fixed_base'):
        plt.title('fixed_base')
    plt.show()



def main(nlist=None, pdir_clf=None, pdir_det=None):
    if nlist == None:
        nlist = []
        for p in os.listdir(pdir_clf):
            if p.endswith('_val'):
                nlist.append(p)
    nlist.sort()
    result = dict()
    for net in nlist:
        net = net[:-4] # - '_val'
        data = namedtuple('result', ['clf', 'det'])
        data.clf = data_extraction_clf(net, pdir_clf)
        data.det = data_extraction_det(net, pdir_det)
        if data.clf == -1 or data.det == -1:
            print('missing {}'.format(net))
            continue
        result[net] = data
    data_plot(result, pdir_det)


if __name__ == '__main__':
    main(pdir_clf='../../examples/imagenet/val', pdir_det='output/ImgNet_pre/train_all')
    