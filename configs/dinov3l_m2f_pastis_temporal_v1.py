custom_imports = dict(
    imports=[
        'custom_datasets.pastis_temporal',
        'custom_models.dinov3_temporal_backbone',
    ],
    allow_failed_imports=False,
)

_base_ = [
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py',
]

DINO_CKPT   = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
PASTIS_ROOT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'

num_classes     = 19
img_size        = 256
in_bands        = 10
n_frames        = 12
total_channels  = n_frames * in_bands   # 120

data_preprocessor = dict(
    _delete_=True,
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32),
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='DINOv3TemporalBackbone',
        arch='vit_large',
        patch_size=16,
        checkpoint=DINO_CKPT,
        interaction_indexes=[5, 11, 17, 23],
        freeze_backbone=False,
        in_bands=in_bands,
        n_frames=n_frames
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1],
        ),
    ),
)

train_pipeline = [
    dict(type='LoadPASTISRasterTemporal', img_size=img_size, num_classes=num_classes),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PASTISRandomRotate90', prob=0.75),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadPASTISRasterTemporal', img_size=img_size, num_classes=num_classes),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='PASTISRasterTemporalDataset',
        data_root=PASTIS_ROOT,
        split='train',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='PASTISRasterTemporalDataset',
        data_root=PASTIS_ROOT,
        split='val',
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 修复：仅对ViT本身（预训练权重）缩小学习率，保证Adapter正常收敛
            'backbone.adapter.backbone': dict(lr_mult=0.05, decay_mult=1.0),
            'backbone.adapter.backbone.patch_embed': dict(lr_mult=1.0, decay_mult=1.0),
            'backbone.adapter.spm.stem': dict(lr_mult=1.0, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat':  embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0, power=0.9, begin=1500, end=20000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=2000, save_best='mIoU', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
)