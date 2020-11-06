# Dual-Refinement-Feature-Pyramid-Networks-for-Object-Detection
---------------------

## Our Development Environment

- Linux (tested on Ubuntu 16.04)
- mmdetection 2.0
- Python 3.7
- PyTorch 1.4
- Cython
- mmcv == 0.6.2

Training
--------------
```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --validate --work_dir <WORK_DIR>
```
For example, our models use 3 RTX Titan GPUs for training
```shell
./tools/dist_train.sh configs/faster_rcnn_r50_drfpn_1x.py 3
```

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection).



## Results on MS COCO test-dev2017

| Backbone | detector | lr schedule| mAP(det) | mAP(mask)  |
|----------|--------|------|-----------|-----------|
| ResNet-50 DRFPN | Faster R-CNN |1x | 39.6 | - |
| ResNet-101 DRFPN | Faster R-CNN |1x | 41.1 | - |
| ResNet-101 DRFPN | Faster R-CNN |2x | 41.8 | - |
| ResNeXt-64x4d-101 DRFPN | Faster R-CNN |1x | 43.4 | - |
| ResNet-50 DRFPN | Mask R-CNN |1x| 40.0 |  36.3 |
| ResNet-101 DRFPN | Mask R-CNN |1x| 41.9 |  37.8 |
| ResNet-101 DRFPN | Mask R-CNN |2x| 42.7 |  38.4 |
| ResNet-50 DRFPN | Cascade Mask R-CNN |1x| 43.0 |  37.7 |
| ResNet-101 DRFPN | Cascade Mask R-CNN |1x| 44.5 |  38.8 |
| ResNet-50 DRFPN| RetinaNet | 1x |  38.0 | -  |
| ResNet-50 DRFPN| FCOS |1x |  37.9  | -  |

## Results on MS COCO val2017

| Backbone | detector | lr schedule| mAP(det) | mAP(mask)  |
|----------|--------|------|-----------|-----------|
| ResNet-50 DRFPN | Faster R-CNN |1x | 39.1 | - |
| ResNet-101 DRFPN | Faster R-CNN |1x | 40.8 | - |
| ResNet-101 DRFPN | Faster R-CNN |2x | 41.5 | - |
| ResNeXt-64x4d-101 DRFPN | Faster R-CNN |1x | 43.0 | - |
| ResNet-50 DRFPN | Mask R-CNN |1x| 39.7 |  35.9 |
| ResNet-101 DRFPN | Mask R-CNN |1x| 41.7 |  37.4 |
| ResNet-101 DRFPN | Mask R-CNN |2x| 42.4 | 38.5  |
| ResNet-50 DRFPN | Cascade Mask R-CNN |1x| 42.8 | 37.4 |
| ResNet-101 DRFPN | Cascade Mask R-CNN |1x| 44.1 | 38.3 |
| ResNet-50 DRFPN| RetinaNet | 1x | 37.6 | -  |
| ResNet-50 DRFPN| FCOS |1x | 37.6  | -  |
