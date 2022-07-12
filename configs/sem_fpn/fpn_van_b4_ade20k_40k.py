_base_ = './fpn_van_b2_ade20k_40k.py'

# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 6, 40, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b4.pth'),
        drop_path_rate=0.4),
    neck=dict(in_channels=[64, 128, 320, 512]))
data = dict(samples_per_gpu=4)
