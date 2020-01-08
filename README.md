# Faster rcnn with various backbones
## backbones
Training and validation for [faster-rcnn](https://arxiv.org/abs/1506.01497).

## build envirionment
Remove 'build', 'faster_rcnn.egg-info' in lib, and remove '_C.cpython-36m-x86_64-linux-gnu.so' in lib/model (if any)
- `cd lib && python setup.py build develop`
- `mkdir data && cd data && ln -s [coco_source_dir] coco`
