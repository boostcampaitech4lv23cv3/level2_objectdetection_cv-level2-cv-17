# dataset settings
dataset_type = "CocoDataset"
data_root = "/opt/ml/dataset/"

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


heavy_aug_default = [
    dict(type="HorizontalFlip", p=0.5),
    dict(type="VerticalFlip", p=0.5),
    dict(
        type="GaussNoise",
        var_limit=(10.0, 50.0),
        mean=0,
        per_channel=True,
        always_apply=False,
        p=0.5,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5,
    ),
    dict(type="RGBShift", r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        border_mode=4,
        rotate_method="largest_box",  # or ellipse
        p=0.5,
    ),
]

# horizontalflip, verticalflip, GaussNoise, RandomBrightnessContrast
# HueSaturationValue, RGBShift, ShiftScaleRotate


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    # dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Albu",
        transforms=heavy_aug_default,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]


test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "val.json",
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

evaluation = dict(interval=1, metric="bbox", classwise=True)
