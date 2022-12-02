# optimizer
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)  # grad_clip : loss가 튀는 경우가 많아 grad_clip을 걸어야할지 말지 정해야함

# learning policy
lr_config = dict(
    policy="step",  # consine annealing 등등..
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
