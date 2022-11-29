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

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
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
        ann_file=data_root + "train_2022_0.8.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    # train=dict(
    #     type='RepeatDataset',
    #     times=2,
    #     dataset=dict(
    #         classes=classes,
    #         type=dataset_type,
    #         ann_file=data_root + "train_2022_0.8.json",
    #         img_prefix=data_root,
    #         pipeline=train_pipeline
    #     )
    # ),
    # cityscapes_detection.py
    # train=dict(
    #     type='RepeatDataset',
    #     times=8,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root +
    #         'annotations/instancesonly_filtered_gtFine_train.json',
    #         img_prefix=data_root + 'leftImg8bit/train/',
    #         pipeline=train_pipeline)),
    # lvis_v1_instance.py
    # train=dict(
    #     _delete_=True,
    #     type='ClassBalancedDataset',
    #     oversample_thr=1e-3,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'annotations/lvis_v1_train.json',
    #         img_prefix=data_root)),
    # voc0712.py
    # train=dict(
    #     type='RepeatDataset',
    #     times=3,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=[
    #             data_root + 'VOC2007/ImageSets/Main/trainval.txt',
    #             data_root + 'VOC2012/ImageSets/Main/trainval.txt'
    #         ],
    #         img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
    #         pipeline=train_pipeline)),
    # wider_face.py
    # train=dict(
    #     type='RepeatDataset',
    #     times=2,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'train.txt',
    #         img_prefix=data_root + 'WIDER_train/',
    #         min_size=17,
    #         pipeline=train_pipeline)),
    # openimages_detection.py
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/oidv6-train-annotations-bbox.csv',
    #     img_prefix=data_root + 'OpenImages/train/',
    #     label_file=data_root + 'annotations/class-descriptions-boxable.csv',
    #     hierarchy_file=data_root +
    #     'annotations/bbox_labels_600_hierarchy.json',
    #     pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + "val_2022_0.2.json",
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
