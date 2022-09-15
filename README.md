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
```

## Results

**Notes**: Pre-trained models can be found in [TsingHua Cloud](https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/).

### VAN + UperNet

|   Method  |    Backbone     |  Pretrained | Iters | mIoU(ms) | Params | FLOPs  | Config | Download  |
| :-------: | :-------------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  UperNet  |    VAN-B0  | IN-1K | 160K | 41.1 | 32M | - | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/1k_pretrained/upernet_van_b0_512x512_160k_ade20k.py)  | - |
|  UperNet  |    VAN-B1  | IN-1K  | 160K |  44.9  | 44M | - | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/1k_pretrained/upernet_van_b1_512x512_160k_ade20k.py)  | - |
|  UperNet  |    VAN-B2  | IN-1K  | 160K |  50.1 | 57M | 948G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/1k_pretrained/upernet_van_b2_512x512_160k_ade20k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/68c8b494f3824d30bf07/?dl=1) |
|  UperNet  |    VAN-B3  | IN-1K  | 160K |  50.6 | 75M | 1030G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/1k_pretrained/upernet_van_b3_512x512_160k_ade20k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/97bde65fbe334b358797/?dl=1) |
|  UperNet  |    VAN-B4  | IN-1K  | 160K |  52.2 |  90M | 1098G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/1k_pretrained/upernet_van_b4_512x512_160k_ade20k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/5273f92c77a94395b804/?dl=1) |
|  UperNet  |    VAN-B4  | IN-22K | 160K |  53.5 | 90M  | 1098G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/22k_pretrained/upernet_van_b4_512x512_160k_ade20k_22k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/8f1f0a9c4c71478fa43b/?dl=1) |
|  UperNet  |    VAN-B5  | IN-22K | 160K |  53.9 |  117M | 1208G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/22k_pretrained/upernet_van_b5_512x512_160k_ade20k_22k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/2175bdc39d094e5f8f99/?dl=1) |
|  UperNet  |    VAN-B6  | IN-22K | 160K |  54.7 | 231M | 1658G | [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/upernet/22k_pretrained/upernet_van_b6_512x512_160k_ade20k_22k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/853d9d0ea0f44c2aa090/?dl=1) |

**Notes**: In this scheme, we use multi-scale validation following Swin-Transformer. FLOPs are tested under the input size of 2048 $\times$ 512 using [torchprofile](https://github.com/zhijian-liu/torchprofile) (recommended, highly accurate and automatic MACs/FLOPs statistics).

### VAN + Semantic FPN

|    Backbone     | Iters | mIoU | Config | Download  |
| :-------------: | :-----: | :------: | :------------: | :----: |
|    VAN-Tiny     | 40K | 38.5 |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/fpn_van_b0_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Jl8LtyvOl6xeNMKCjpK2Rp_tGRfua8LJ/view?usp=sharing) |
|    VAN-Small    | 40K |  42.9  |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/fpn_van_b1_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Xfuo9D3Fo7b6zSCLTWE77k2jgYSHVSb8/view?usp=sharing) |
|    VAN-Base     | 40K | 46.7   |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/fpn_van_b2_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1Ar4Hq9DjgaULQKfwM-jJvSO-D6gendpf/view?usp=sharing) |
|    VAN-Large    | 40K |  48.1  |  [config](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/configs/sem_fpn/fpn_van_b3_ade20k_40k.py)  | [Google Drive](https://drive.google.com/file/d/1v61uCi07IC6eyVHn3xbJqz4nOiGa1POY/view?usp=sharing) |

## Preparation

Install MMSegmentation and download ADE20K according to the guidelines in MMSegmentation.

## Requirement

```
pip install mmsegmentation==0.26.0 (https://github.com/open-mmlab/mmsegmentation/tree/v0.26.0)
```

## Training

We use 8 GPUs for training by default. Run:

```bash
./dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the model, run:

```bash
./dist_test.sh /path/to/config /path/to/checkpoint_file 8 --eval mIoU
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/flops.sh /path/to/config --shape 512 512
```


## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0), [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
