# Visual Attention Network (VAN) for Semantic Segmentaion
This repo is a PyTorch implementation of applying **VAN** (**Visual Attention Network**) to semantic segmentation.
The code is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0).

More detailes can be found in 

### Results
**Notes**: Pre-trained models can be found in [Visual Attention Network for classification](https://github.com/Visual-Attention-Network/VAN-Classification)
#### VAN + UperNet
|    Backbone     | Iters | mIoU | Config | Download  |
| :-------------: | :-----: | :------: | :------------: | :----: |
|    VAN-Tiny     | 160K | 41.1 |  [config]()  |  |
|    VAN-Small    | 160K |  44.9  |  [config]()  |  |
|    VAN-Base     | 160K | 48.3   |  [config]()  |  |
|    VAN-Large    | 160K |  50.1  |  [config]()  |  |
**Notes**: In this scheme, we use multi-scale validation following Swin-Transformer.

#### VAN + Semantic FPN
|    Backbone     | Iters | mIoU | Config | Download  |
| :-------------: | :-----: | :------: | :------------: | :----: |
|    VAN-Tiny     | 40K | 38.5 |  [config]()  |  |
|    VAN-Small    | 40K |  42.9  |  [config]()  |  |
|    VAN-Base     | 40K | 46.7   |  [config]()  |  |
|    VAN-Large    | 40K |  48.1  |  [config]()  |  |


### Preparation
Install MMSegmentation and download ADE20K according to the guidelines in MMSegmentation.

### Training
We use 8 GPUs for training by default. Run:
```bash
dist_train.sh /path/to/config 8
```
### Evaluation
To evaluate the model, run:
```bash
dist_test.sh /path/to/config /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```

## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0), [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), and [PoolFormer](https://github.com/sail-sg/poolformer). Thanks for their authors. 

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.


