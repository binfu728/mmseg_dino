custom_imports = dict(
    imports = ['custom_models.dinov3_backbone','custom_datasets.pastis'],
    allow_failed_imports = False
)

# ── 1. 继承官方的 FCN 模型基础结构和默认运行环境 ──────────────────────────
_base_ = [
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/models/fcn_r50-d8.py',
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/default_runtime.py'
]

# ── 2. Paths to set manually ──────────────────────────────────────────────────
DINO_CKPT   = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
PASTIS_ROOT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'

num_classes = 18    # crop classes 0-17
img_size    = 512   # 128×128 raster patch upsampled to 512×512 for ViT

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[1436.7, 1387.7, 1180.2],
    std=[1996.2, 1916.8, 1976.7],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32),
)

# ── 3. 覆盖模型结构 (只写需要修改和替换的部分) ───────────────────────────────
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,             # 关键：删掉 _base_ 里的 ResNet50
        type='DINOv3BackboneMmseg',
        arch='vit_small',          # vit_small: embed_dim=384
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=False,      # True=linear probe; False=full finetune
    ),
    decode_head=dict(
        in_channels=384,           # 覆盖 _base_ 里的 2048，适配 DINOv3-S
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0] * num_classes + [0.1],  # PASTIS 类别权重
        ),
    ),
    auxiliary_head=dict(
        in_channels=384,           # 覆盖 _base_ 里的 1024，适配 DINOv3-S
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,       # 辅助头 loss 权重按官方保持 0.4
            class_weight=[1.0] * num_classes + [0.1],
        ),
    ),
)

# ── 4. 数据集与 Pipeline ──────────────────────────────────────────────────────
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
    sampler=dict(type='InfiniteSampler', shuffle=True),
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

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ── 5. 优化器与训练策略 ───────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# 覆盖 _base_/default_runtime.py 里的保存策略
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
)