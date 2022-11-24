# optimizer
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)                     # grad_clip : loss가 튀는 경우가 많아 grad_clip을 걸어야할지 말지 정해야함
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
)

momentum_config = dict(
    policy="cyclic",
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

runner = dict(type="EpochBasedRunner", max_epochs=12)
