# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/data/mmseg/'


classes = [
    'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
palette = [
    [0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128],
    [64, 0, 192], [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
]

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    # dict(type="RandomRotate90", p=0.5),
    # dict(type="GridDropout", ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomRotate90', p=1.0),
            dict(type='Rotate', limit=(30,30), p=1.0),
            dict(type='Rotate', limit=(-30,-30), p=1.0),
        ], p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomBrightnessContrast", brightness_limit=0.25, contrast_limit=0.25),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="GaussNoise"),          
        ],
        p=0.5,
    )
]

crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='Resize',
    #     img_scale=[(256,256),(512,512)],
    #     multiscale_mode = 'value',
    #     keep_ratio=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_root + "images/training",
        ann_dir=data_root + "annotations/training",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_root + "test",
        pipeline=test_pipeline))