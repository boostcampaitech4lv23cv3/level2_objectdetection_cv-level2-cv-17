_base_ = ["../_base_/models/retinanet_r50_fpn.py", "../common/mstrain_3x_coco_alb.py"]
# optimizer
model = dict(
    backbone=dict(
        type="ResNeXt",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
        depth=101,
        groups=64,
        base_width=4,
    )
)

optimizer = dict(type="SGD", lr=0.001)
