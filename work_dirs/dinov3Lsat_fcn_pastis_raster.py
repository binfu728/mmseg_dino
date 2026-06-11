DINO_LARGE_CKPT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
PASTIS_ROOT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'custom_models.dinov3_backbone',
        'custom_datasets.pastis',
    ])
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        1436.7,
        1387.7,
        1180.2,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        1996.2,
        1916.8,
        1976.7,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, save_best='mIoU', type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=True, type='LoggerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_size = 512
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=18,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        arch='vit_large',
        checkpoint=
        '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=False,
        patch_size=16,
        type='DINOv3BackboneMmseg'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            1436.7,
            1387.7,
            1180.2,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            1996.2,
            1916.8,
            1976.7,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        concat_input=True,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=18,
        num_convs=2,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 18
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        cooldown=10,
        factor=0.2,
        monitor='mIoU',
        patience=2,
        rule='greater',
        type='ReduceOnPlateauLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root='/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R',
        pipeline=[
            dict(img_size=512, type='LoadPASTISRaster'),
            dict(type='PackSegInputs'),
        ],
        split='val',
        type='PASTISRasterDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root='/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R',
        pipeline=[
            dict(img_size=512, type='LoadPASTISRaster'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        split='train',
        type='PASTISRasterDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(img_size=512, type='LoadPASTISRaster'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root='/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R',
        pipeline=[
            dict(img_size=512, type='LoadPASTISRaster'),
            dict(type='PackSegInputs'),
        ],
        split='val',
        type='PASTISRasterDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
val_pipeline = [
    dict(img_size=512, type='LoadPASTISRaster'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/'
