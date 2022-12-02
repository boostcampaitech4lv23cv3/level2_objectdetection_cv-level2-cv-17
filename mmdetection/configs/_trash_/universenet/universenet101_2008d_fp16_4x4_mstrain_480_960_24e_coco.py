_base_ = ["../_base_/default_runtime.py"]

# model settings
model = dict(
    type="GFL",
    backbone=dict(
        type="Res2Net",
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://res2net101_v1d_26w_4s"
        ),
    ),
    # checkpoint=pretrained)),
    neck=[
        dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
        ),
        dict(
            type="SEPC",
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=True,  # please set imgs/gpu >= 4
            pnorm_eval=False,
            lcnorm_eval=False,
            lcconv_padding=1,
        ),
    ],
    bbox_head=dict(
        type="GFLSEPCHead",
        num_classes=10,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_dfl=dict(type="DistributionFocalLoss", loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

data = dict(samples_per_gpu=16)

optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)

# dataset settings
dataset_type = "CocoDataset"
data_root = "../../dataset/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

albu_train_transforms = [
    dict(type="Emboss", alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
    dict(type="Sharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    dict(type="CLAHE", clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    dict(type="VerticalFlip", p=0.5),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="RandomRotate90", p=0.3),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="RGBShift",
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0,
            ),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
        ],
        p=0.1,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MedianBlur", blur_limit=3, p=1.0),
        ],
        p=0.1,
    ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size_divisor=32),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# Use RepeatDataset to speed up training
classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train_groupk_3.json",
        classes=classes,
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_groupk_3.json",
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")

# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=1000, warmup_ratio=0.001, step=[16, 19]
)

runner = dict(type="EpochBasedRunner", max_epochs=24)
