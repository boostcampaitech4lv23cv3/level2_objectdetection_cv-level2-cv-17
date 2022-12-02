_base_ = "./queryinst_r50_fpn_mstrain_480-800_3x_coco.py"

model = dict(
    backbone=dict(
        type="ResNeXt",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
        depth=101,
        groups=64,
        base_width=4,
    )
)
