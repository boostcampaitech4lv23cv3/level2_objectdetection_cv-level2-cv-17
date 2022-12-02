_base_ = "./queryinst_r50_fpn_1x_coco.py"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)

lr_config = dict(policy="step", step=[27, 33])
runner = dict(type="EpochBasedRunner", max_epochs=36)
