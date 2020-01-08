# Faster rcnn with various backbones
## backbones
Training and validation for [faster-rcnn](https://arxiv.org/abs/1506.01497).

## build envirionment
- torch >= 1.1
- torchvision >= 0.3

Remove 'build', 'faster_rcnn.egg-info' in `lib`, and remove '_C.cpython-36m-x86_64-linux-gnu.so' in `lib/model` (if any)
- `cd lib && python setup.py build develop`
- `mkdir data && cd data && ln -s [coco_source_dir] coco`

## ImageNet accuracy with pretrained models in pytorch
Evaluation results for ImageNet validation split (5000 images). 

| Backbone | Acc@1 | Acc@5 |
| -------- | ----- | ----- |
| vgg13 | 69.923 | 89.246 |
| vgg16 | 71.592 | 90.382 |
| vgg19 | 72.376 | 90.876 |
| resnet18 | 69.758 | 89.078 |
| resnet34 | 73.314 | 91.420 |
| resnet50 | 76.130 | 92.862 |
| resnet101 | 77.374 | 93.546 |
| resnet152 | 78.312 | 94.046 |
