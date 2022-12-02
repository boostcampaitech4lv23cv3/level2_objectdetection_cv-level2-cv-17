# dataset settings
dataset_type = "CocoDataset"
data_root = "/opt/ml/dataset/"

"""
    Heavy Augmenteation + TTA(RandomFlip)
"""
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

img_scale = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
albu_train_transforms = [
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2,
    ),
    # dict(
    #     # 입체감을 주며 더욱 선명해지는 효과
    #     type="Emboss",
    #     alpha=[0.2, 0.5],
    #     strength=[0.2, 0.7],
    #     p=0.5,
    # ),
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
    dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.2),
    # dict(type="ChannelShuffle", p=0.1),
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
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    # dict(  # multi-scale
    #     type="Resize",
    #     img_scale=[
    #         (1024, 1024),
    #         (512, 512),
    #         (256, 256),
    #     ],
    #     multiscale_mode="value",
    #     keep_ratio=True,
    # ),
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
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            # dict(type="RandomFlip"),
            dict(type="RandomFlip", flip_ratio=0.5, direction="vertical"),
            dict(type="RandomFlip", flip_ratio=0.5, direction="horizontal"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "train_groupk_3.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "val_groupk_3.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric="bbox")
