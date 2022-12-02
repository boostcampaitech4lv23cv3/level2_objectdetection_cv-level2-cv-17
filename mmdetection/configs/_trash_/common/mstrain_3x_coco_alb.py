_base_ = "../_base_/default_runtime.py"
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
# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train_groupk_1.json",
            classes=classes,
            img_prefix=data_root,
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_groupk_1.json",
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
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[9, 11]
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
