# https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="CheckpointHook"),
        dict(
            type="TensorboardLoggerHook",
            log_dir="",
            interval=10,
            ignore_last=True,
            reset_flag=False,
            by_epoch=True,
        ),
        # Args:
        #     log_dir (string): Save directory location. Default: None. If default
        #         values are used, directory location is ``runner.work_dir``/tf_logs.
        #     interval (int): Logging interval (every k iterations). Default: True.
        #     ignore_last (bool): Ignore the log of last iterations in each epoch
        #         if less than `interval`. Default: True.
        #     reset_flag (bool): Whether to clear the output buffer after logging.
        #         Default: False.
        #     by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        # dict(type='MlflowLoggerHook'),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": "object-detection",
                "entity": "boostcamp-ai-tech-4-cv-17",
            },
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3,
        ),
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None

workflow = [("train", 1)]
# https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow
# workflow = [("train", 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
