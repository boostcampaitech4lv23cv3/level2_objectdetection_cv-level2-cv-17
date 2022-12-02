# model settings
model = dict(
    type="RetinaNet",
    backbone=dict(type="EfficientNet", model_name="tf_efficientnet_b4"),
    neck=dict(
        type="BIFPN",
        in_channels=[56, 112, 160, 272, 448],
        out_channels=224,
        start_level=0,
        stack=6,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=dict(type="BN", requires_grad=False),
        activation="relu",
    ),
    bbox_head=dict(
        type="RetinaHead",
        num_classes=81,
        in_channels=224,  # 256->224
        stacked_convs=4,
        feat_channels=224,  # 256->224
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=1.5,  # 2->1.5
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_thr=0.5),
        max_per_img=100,
    ),
)

# work_dir = "./work_dirs/efficient_d4_bifpn_1x"
