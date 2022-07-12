_base_ = './fpn_van_b2_ade20k_40k.py'

# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b1.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
