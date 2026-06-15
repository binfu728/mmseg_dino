# DINOv3 ViT-S + Mask2Former on PASTIS-R, Phase 1+2 combined
# (docs_zf/dinov3_pastis_mIoU提升方案.md, experiments E1+E3):
#   * 10 spectral bands, stems inflated 3->10 channels
#   * 12 monthly median composite frames, per-frame encoding + temporal mean pool
#   * U-TAE protocol: 19 classes (background = class 0), only void ignored
#   * backbone unfrozen (lr x0.1), warmup + poly, flip + rot90 augmentation, AMP
#
# Launch: bash jzf/train.sh   (needs PYTHONPATH=<mmsegmentation root> for jzf.*)

_base_ = ['../../configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(
    imports=['jzf.pastis_temporal', 'jzf.dinov3_temporal_backbone'],
    allow_failed_imports=False)

DINO_CKPT = ('/home/zifei/.cache/modelscope/hub/models/facebook/dinov3pth/'
             'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
PASTIS_ROOT = '/home/zifei/dataset/PASTIS-R'

num_classes = 19   # background(0) + 18 crops; void(19) -> 255
img_size = 256     # 128 native x2; ViT-S/16 -> 16x16 tokens per frame
n_frames = 12
in_bands = 10

# Normalisation happens inside LoadPASTISRasterTemporal; preprocessor only
# stacks/pads.
data_preprocessor = dict(
    _delete_=True,
    type='SegDataPreProcessor',
    mean=None, std=None,
    bgr_to_rgb=False, pad_val=0, seg_pad_val=255,
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='DINOv3TemporalBackbone',
        arch='vit_small',
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=False,
        in_bands=in_bands,
        n_frames=n_frames,
        temporal_agg='mean'),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=2.0, reduction='mean',
            class_weight=[1.0] * num_classes + [0.1])))

train_pipeline = [
    dict(type='LoadPASTISRasterTemporal', img_size=img_size, n_frames=n_frames),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PASTISRandomRotate90', prob=0.75),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadPASTISRasterTemporal', img_size=img_size, n_frames=n_frames),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4, num_workers=4,
    dataset=dict(
        _delete_=True, type='PASTISRasterTemporalDataset',
        data_root=PASTIS_ROOT, split='train', pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        _delete_=True, type='PASTISRasterTemporalDataset',
        data_root=PASTIS_ROOT, split='val', pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',  # fp16 overflows in M2F mask BCE (loss_mask=nan); bf16 is stable
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter.backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))

max_iters = 20000
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
