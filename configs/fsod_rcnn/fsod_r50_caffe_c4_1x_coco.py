_base_ = [
    '../_base_/models/fsod_r50_caffe_c4.py',
    '../_base_/datasets/coco_detection_fsod.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False) # pls do not change
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,with_id=True ),
    dict(type='SelectOne', seed = 0),
    dict(type='LoadSupport', num_imgs = 10, support_df = 'train_support_df.pkl'), #
    dict(type='LoadSupportImageFromFile'),
    dict(type='LabelToZero'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg), #changed
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),  # changed
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'support_imgs', 'support_bboxes', 'support_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img',]),
        ])
]
# custom_imports = dict(imports=['mmdet.core.utils.fsod_hook'], allow_failed_imports=False)
# custom_hooks = [
#     dict(type='Hook_fsod', a=1, b=2, priority='HIGHEST')
# ]
data = dict(
    workers_per_gpu=0, #! 2 for debug data pipeline
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# workflow = [('test', 1)]
