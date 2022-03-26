# Visual Attention Network (VAN) for Segmentaion

This repo is a PyTorch implementation of applying **VAN** (**Visual Attention Network**) to semantic segmentation.
The code is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0).

More details can be found in [**Visual Attention Network**](https://arxiv.org/abs/2202.09741).

## Citation

```bib
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}

@inproceedings{
    ham,
    title={Is Attention Better Than Matrix Decomposition?},
    author={Zhengyang Geng and Meng-Hao Guo and Hongxu Chen and Xia Li and Ke Wei and Zhouchen Lin},
    booktitle={International Conference on Learning Representations},
    year={2021},
}
```

## Results

**Notes**: Pre-trained models can be found in [Visual Attention Network for Classification](https://github.com/Visual-Attention-Network/VAN-Classification).

### VAN + Light-Ham / HamNet / UperNet

|   Method  |    Backbone     | Iters | mIoU | Params | FLOPs  | Config | Download  |
| :-------: | :-------------: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  Light-Ham-D256  |    VAN-Tiny     | 160K | [40.9](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/work_dirs/hamnet_light_van_tiny_d256_512x512_160k_ade20k/eval_multi_scale_20220321_052101.json) | 4.2M | 6.5G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/ham/hamnet_light_van_tiny_d256_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/11XjGgqgqWJOUKdIEWuInQJyi4wAChaWN/view?usp=sharing) |
|  Light-Ham  |    VAN-Tiny     | 160K | [42.3](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/work_dirs/hamnet_light_van_tiny_512x512_160k_ade20k/eval_multi_scale_20220323_130645.json) | 4.9M | 11.3G |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/ham/hamnet_light_van_tiny_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1MkjNxqOuoVtt58jIKY-11B6TfDrdH1sX/view?usp=sharing) |
|  Light-Ham  |    VAN-Small    | 160K | [45.7](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/work_dirs/hamnet_light_van_small_512x512_160k_ade20k/eval_multi_scale_20220323_124229.json) | 14.7M | 21.4G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/ham/hamnet_light_van_small_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1ocFNvV2Dr8kXsytY_9QO5FBGk1zTOgS3/view?usp=sharing) |
|  Light-Ham  |    VAN-Base     | 160K | [49.6](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/work_dirs/hamnet_light_van_base_512x512_160k_ade20k/eval_multi_scale_20220323_135751.json) | 27.4M | 34.4G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/ham/hamnet_light_van_base_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1-wVJgdztqWYv-MvCp6deFO0pDpciAg6h/view?usp=sharing) |
|  Light-Ham  |    VAN-Large    | 160K | [51.0](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/work_dirs/hamnet_light_van_large_512x512_160k_ade20k/eval_multi_scale_20220323_142104.json) | 45.6M | 55.0G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/ham/hamnet_light_van_large_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1iW-upuWcZybJyGv8_3qnpgGoX0Wq9emk/view?usp=sharing) |
|  -  | - | - | - | -  | - | - | - |
|  HamNet  |    VAN-Tiny-OS8     | 160K | 41.5 | 11.9M | 50.8G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_tiny_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1T1BxnBr4rErKaKiUwp_xF-Ik7j7jINJR/view?usp=sharing) |
|  HamNet  |    VAN-Small-OS8    | 160K | 45.1 | 24.2M | 100.6G |   [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_small_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1kfZIMZINOprSL6G113sm_KjPlE10nbWz/view?usp=sharing) |
|  HamNet  |    VAN-Base-OS8     | 160K | 48.7 | 36.9M | 153.6G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_base_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1jH1jx6KPckEL0-Ozje0koT8uFw0Bjyfi/view?usp=sharing) |
|  HamNet  |    VAN-Large-OS8    | 160K | 50.2 | 55.1M | 227.7G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_large_512x512_160k_ade20k.py)  | [Google Drive](https://drive.google.com/file/d/1tPEQ9W1Pn_Bmkn3eGOtjM8dMZ0mTK4ka/view?usp=sharing) |
|  -  | - | - | - | -  | - | - | - |
|  UperNet  |    VAN-Tiny  | 160K | 41.1 | 32.1M | 214.7 | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_tiny_512x512_160k_ade20k.py)  |  |
|  UperNet  |    VAN-Small    | 160K |  44.9  | 43.8M | 224.0G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_small_512x512_160k_ade20k.py)  |  |
|  UperNet  |    VAN-Base     | 160K | 48.3 | 56.6M | 237.1G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_base_512x512_160k_ade20k.py)  |  |
|  UperNet  |    VAN-Large    | 160K |  50.1 | 74.7M | 257.7G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/VAN/upernet_van_large_512x512_160k_ade20k.py)  |  |

**Notes**: In this scheme, we use multi-scale validation following Swin-Transformer. FLOPs are tested under the input size of 512 $\times$ 512 using [torchprofile](https://github.com/zhijian-liu/torchprofile) (recommended, highly accurate and automatic MACs/FLOPs statistics).

### VAN + Semantic FPN

|    Backbone     | Iters | mIoU | Config | Download  |
| :-------------: | :-----: | :------: | :------------: | :----: |
|    VAN-Tiny     | 40K | 38.5 |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/VAN/fpn_van_tiny_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Jl8LtyvOl6xeNMKCjpK2Rp_tGRfua8LJ/view?usp=sharing) |
|    VAN-Small    | 40K |  42.9  |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/VAN/fpn_van_small_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Xfuo9D3Fo7b6zSCLTWE77k2jgYSHVSb8/view?usp=sharing) |
|    VAN-Base     | 40K | 46.7   |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/VAN/fpn_van_base_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Ar4Hq9DjgaULQKfwM-jJvSO-D6gendpf/view?usp=sharing) |
|    VAN-Large    | 40K |  48.1  |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/VAN/fpn_van_large_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1v61uCi07IC6eyVHn3xbJqz4nOiGa1POY/view?usp=sharing) |

## Preparation

Install MMSegmentation and download ADE20K according to the guidelines in MMSegmentation.

## Requirement

```
Pytorch >= 1.7
MMSegmentation == v0.12.0 (https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0)
```

## Training

We use 8 GPUs for training by default. Run:

```bash
dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the model, run:

```bash
dist_test.sh /path/to/config /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/flops.sh /path/to/checkpoint_file --shape 512 512
```


## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0), [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
