"""DINOv3-Large + FCN 全参微调 — OlmoEarth 时间特征池化方案.

核心改造 (OlmoEarth Section 3.3):
    1. 输入: 12 时相 RGB 拼接为 36 通道伪图像 (H, W, 36)
    2. Backbone: 展开时间→ViT 提取→特征层 Mean Pooling (OlmoEarth 核心)
    3. Decoder: FCN head, 单层输出 (无 auxiliary_head)

与 dinov3Lsat_fcn_pastis_raster.py 的区别:
    - 使用 DINOv3BackboneMmsegTemporal (纯 ViT, 无 adapter)
    - 36 通道 normalization (vs 3 通道)
    - img_size=224, batch_size=4 (时间展开后有效 batch=48)
    - 无 auxiliary_head (单输出 backbone)
"""

custom_imports = dict(
    imports=['custom_models.dinov3_backbone_temporal', 'custom_datasets.pastis'],
    allow_failed_imports=False
)

_base_ = [
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/models/fcn_r50-d8.py',
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/default_runtime.py'
]

DINO_LARGE_CKPT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
PASTIS_ROOT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'

num_classes = 19
img_size = 224

# 36 通道 Normalization: 12 个时相 × 3 通道 RGB, 每通道独立归一化
_PASTIS_RASTER_RGB_MEAN = [1436.7, 1387.7, 1180.2]
_PASTIS_RASTER_RGB_STD = [1996.2, 1916.8, 1976.7]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=_PASTIS_RASTER_RGB_MEAN * 12,
    std=_PASTIS_RASTER_RGB_STD * 12,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32),
)

model = dict(
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='DINOv3BackboneMmsegTemporal',
        arch='vit_large',
        patch_size=16,
        checkpoint=DINO_LARGE_CKPT,
        freeze_backbone=False,
    ),
    decode_head=dict(
        in_channels=1024,  # ViT-Large embed_dim
        in_index=0,        # 单输出 backbone 必须显式指定索引
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    # 单输出 backbone 无法支持 auxiliary_head
    auxiliary_head=None,
)

train_pipeline = [
    dict(type='LoadPASTISRaster', img_size=img_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadPASTISRaster', img_size=img_size),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PASTISRasterDataset',
        data_root=PASTIS_ROOT,
        split='train',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PASTISRasterDataset',
        data_root=PASTIS_ROOT,
        split='val',
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)},
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(
        type='ReduceOnPlateauLR',
        monitor='mIoU',
        rule='greater',
        factor=0.2,
        patience=2,
        cooldown=10,
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
)
