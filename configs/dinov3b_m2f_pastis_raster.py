custom_imports = dict(
    imports = ['custom_models.dinov3_backbone','custom_datasets.pastis'],
    allow_failed_imports = False
)

_base_ = [
    '/mnt/ht2_nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py',
]

# ── Paths to set manually ─────────────────────────────────────────────────────
DINO_CKPT   = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
PASTIS_ROOT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R'   # dir with metadata.geojson, DATA_S2/, ANNOTATIONS/
# ─────────────────────────────────────────────────────────────────────────────

num_classes = 18    # crop classes 0-17; background (orig 0) → ignore_index=255
img_size    = 512   # 128×128 raster patch upsampled to 512×512 for ViT

# Per-channel stats from NORM_S2_patch.json, averaged over all 5 folds, RGB order (B04,B03,B02)
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

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='DINOv3BackboneMmseg',
        arch='vit_small',          # vit_small: embed_dim=384; vit_large: 1024
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=True,      # True=linear probe; False=full finetune
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],   # ViT-S embed_dim, same across 4 scales
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1],   # 18 crops + no-object
        ),
    ),
)

# Real spatial images: RandomFlip is meaningful (unlike PixelSet pseudo-images)
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
    batch_size=4,       # 512×512 images; halved vs PixelSet 64×64 config
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
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
    dataset=dict(
        _delete_=True,
        type='PASTISRasterDataset',
        data_root=PASTIS_ROOT,
        split='val',
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ── Optimiser ─────────────────────────────────────────────────────────────────
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat':  embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

# ── Schedule ──────────────────────────────────────────────────────────────────
param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=40000,
         by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=2000, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
)
