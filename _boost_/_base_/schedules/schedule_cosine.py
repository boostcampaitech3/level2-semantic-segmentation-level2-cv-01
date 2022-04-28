# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # step=[8, 11]
    )
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, classwise=True, save_best='mIoU')