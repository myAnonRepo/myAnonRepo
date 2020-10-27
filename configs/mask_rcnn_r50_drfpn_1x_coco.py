_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    neck=dict(
        type='DRFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
data = dict(samples_per_gpu=8)
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/mask_rcnn_r50_drfpn_1x_coco'
