# Faster rcnn with various backbones
## backbones
Training and validation for [faster-rcnn](https://arxiv.org/abs/1506.01497).

## build envirionment
- torch >= 1.1
- torchvision >= 0.3

Remove 'build', 'faster_rcnn.egg-info' in `lib`, and remove '_C.cpython-~.so' in `lib/model` (if any)
- `cd lib && python setup.py build develop`
- `mkdir data && cd data && ln -s [coco_source_dir] coco`

## ImageNet accuracy with pretrained models in pytorch
Evaluation results for ImageNet validation split. 

| Backbone | Acc@1 | Acc@5 | IoU=0.50:0.95 |
| -------- | ----- | ----- | ----- |
| vgg13 | 69.923 | 89.246 | 0.209 |
| vgg16 | 71.592 | 90.382 | 0.234 |
| vgg19 | 72.376 | 90.876 | 0.243|
| resnet18 | 69.758 | 89.078 ||
| resnet34 | 73.314 | 91.420 ||
| resnet50 | 76.130 | 92.862 | 0.312 |
| resnet101 | 77.374 | 93.546 | 0.338 |
| resnet152 | 78.312 | 94.046 | 0.334 |
| densenet121 | 74.434 | 91.972 ||
| densenet161 | 77.138 | 93.560 ||
| densenet169 | 75.600 | 92.806 ||
| densenet201 | 76.896 | 93.370 ||
| se_resnet50 | 73.990 | 91.774 ||
| se_resnet101 | 76.592 | 93.264 ||
| se_resnet152 | 77.530 | 93.734 ||
| resnext50_32x4d | 77.618 | 93.698 ||
| resnext101_32x8d | 79.312 | 94.526 ||
| mobilenet_v2 | 71.878 | 90.286 ||
| shufflenet_v2_x0_5 | 60.552 | 81.746 ||
| shufflenet_v2_x1_0 | 69.362 | 88.316 ||
