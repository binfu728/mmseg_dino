custom_imports = dict(
    imports = ['custom_models.dinov3_backbone','custom_datasets.pastis'],
    allow_failed_imports = False
)

# ── 1. 继承官方基础结构和默认运行环境 ─────────────────────────────────────────
_base_ = [
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/models/fcn_r50-d8.py',
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/_base_/default_runtime.py'
]

# ── 2. Paths (请修改为你的 Large 模型权重路径) ─────────────────────────────────
DINO_LARGE_CKPT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth' 
PASTIS_ROOT     = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'

num_classes = 19    # crop classes 0-17
img_size    = 512   

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

# ── 3. 模型定义 (ViT-Large 全参微调) ───────────────────────────────────────────
model = dict(
    pretrained=None,                
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,              
        type='DINOv3BackboneMmseg',
        arch='vit_large',           
        patch_size=16,              # 注意：Large 版本通常 patch_size 是 14
        checkpoint=DINO_LARGE_CKPT, 
        freeze_backbone=False,      # 忽略前20%冻结，直接从头开始全参微调
    ),
    decode_head=dict(
        in_channels=1024,           
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            # 保持移除 class_weight 以避开 255 越界 bug
        ),
    ),
    auxiliary_head=dict(
        in_channels=1024,           
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,       
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

# 【硬件适配】A40 48G 显存训练 ViT-Large，稳妥起见 bs=2
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), # 改用 DefaultSampler 适配按 Epoch 训练
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

# ── 5. 优化器与训练策略 (严格对齐 OlmoEarth 论文) ──────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    # 论文设定：AdamW, learning rate = 1e-4
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0), # 给骨干网络稍微小一点的学习率更稳
        },
        norm_decay_mult=0.0,
    ),
)

# 论文设定：Plateau scheduler (验证集不提升时降低LR)
param_scheduler = [
    dict(
        type='ReduceOnPlateauLR',
        monitor='mIoU',        # 监控验证集的 mIoU
        rule='greater',            # mIoU 越大越好
        factor=0.2,            # 降低系数：乘以 0.2
        patience=2,            # 容忍度：2 个 epoch 没有提升就触发下降
        cooldown=10,           # 冷却期：触发下降后的 10 个 epoch 内不再触发
    )
]

# 配合 Plateau，必须采用按 Epoch 训练的循环机制
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# Hook 配置也必须同步修改为按 Epoch 触发
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
)