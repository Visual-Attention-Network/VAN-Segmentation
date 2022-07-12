_base_ = './fpn_van_b2_ade20k_40k.py'

# model settings
model = dict(
    backbone=dict(
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b0.pth')),
    neck=dict(in_channels=[32, 64, 160, 256]))
