# Faster rcnn with various backbones
## backbones
Training and validation for [faster-rcnn](https://arxiv.org/abs/1506.01497).

## build envirionment
- torch <= 1.1
- torchvision <= 0.3

Remove 'build', 'faster_rcnn.egg-info' in `lib`, and remove '_C.cpython-36m-x86_64-linux-gnu.so' in `lib/model` (if any)
- `cd lib && python setup.py build develop`
- `mkdir data && cd data && ln -s [coco_source_dir] coco`

## ImageNet accuracy with pretrained models in pytorch
| Backbone | Acc@1 | Acc@5 |
| -------- | ----- | ----- |
| vgg13 | | |
| vgg16 | | |
| vgg19 | | |
| resnet18 | | |
| resnet34 | | |
| resnet50 | | |
| resnet101 | | |
| resnet152 | | |
