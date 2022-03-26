_base_ = './fpn_van_base_ade20k_40k.py'

# model settings
model = dict(
    pretrained='./seg_pretrained/checkpoint-306.pth.tar',
    backbone=dict(
        type='van_large',
        style='pytorch'),
    neck=dict(in_channels=[64, 128, 320, 512]))
