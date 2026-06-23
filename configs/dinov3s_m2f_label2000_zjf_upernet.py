# DINOv3 ViT-L + UPerNet on SVDT binary cropland segmentation
#   * 3 channel RGB input (SVDT annual composite satellite imagery)
#   * 2 classes: background (0), cropland (1)
#   * backbone unfrozen (lr x0.1), warmup + poly, flip + rot90 augmentation, AMP
#   * DDP distributed training ready (find_unused_parameters=True)
#   * Dataset: SVDTDataset, data at /mnt/ht2_nas2/EO_test/openmmlab-archive/dat/SVDT
#
# Launch (single GPU):
#   python3 train.py configs/dinov3s_m2f_label2000_zjf_upernet.py
# Launch (multi-GPU DDP):
#   torchrun --nproc_per_node=4 train.py configs/dinov3s_m2f_label2000_zjf_upernet.py --launcher pytorch

_base_ = ['/mnt/ht2-nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/upernet/upernet_r50_4xb4-160k_ade20k-512x512.py']

#img_size=256时候的设置

custom_imports = dict(
    imports=['custom_datasets.label20000_temporal_zjf_upernet_v2', 
    'custom_models.dinov3_backbone_label20000_v2'],
    allow_failed_imports=False)

# #img_size=512时候的设置
# custom_imports = dict(
#     imports=['custom_datasets.label20000_temporal_zjf_upernet_v2', 'custom_models.dinov3_backbone_label20000_512_v3'],
#     allow_failed_imports=False)


DINO_CKPT = ('/mnt/ht2-nas2/EO_test/weights/Dinov3_pretrained/DINOv3_ViT_SAT-493M/'
             'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
DATA_ROOT = '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/SVDT'

img_size    = 256
# img_size    = 512
num_classes = 2
max_iters   = 20000

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None, std=None,
    bgr_to_rgb=False, pad_val=0, seg_pad_val=255,
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    _delete_=True,
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOv3BackboneMmseg_hlj',
        arch='vit_large',
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=False,
        in_bands=3),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadSVDT', img_size=img_size),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='SVDTRandomRotate90', prob=0.75),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadSVDT', img_size=img_size),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4, num_workers=4,
    dataset=dict(
        _delete_=True, type='SVDTDataset',
        data_root=DATA_ROOT, split='train', pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        _delete_=True, type='SVDTDataset',
        data_root=DATA_ROOT, split='val', pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter.backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0, power=0.9, begin=1500, end=max_iters,
         by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
                    save_best='mIoU', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))

find_unused_parameters = True
