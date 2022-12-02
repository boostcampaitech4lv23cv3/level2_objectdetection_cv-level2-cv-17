_base_ = "../_base_/default_runtime.py"

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"  # noqa
model = dict(
    type="ATSS",
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=[
        dict(
            type="FPN",
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=5,
        ),
        dict(
            type="DyHead",
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False,
        ),
    ],
    bbox_head=dict(
        type="ATSSHead",
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,  # follow DyHead official implementation
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5,
        ),  # follow DyHead official implementation
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
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
        img_scale=(2000, 1200),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True, backend="pillow"),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=128),
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
    samples_per_gpu=2,  # oom 문제로 2에서 1로 수정
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train_groupk_2.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_groupk_2.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")

# optimizer
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
optimizer = dict(
    type="AdamW",
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=2e-7,
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
