# Read Between the Layers: Leveraging Multi-Layer Representations for Rehearsal-Free Continual Learning with Pre-Trained Models

## Official code repository for our [TMLR 2024](https://jmlr.org/tmlr/papers/) paper:

Kyra Ahrens, Hans Hergen Lehmann, Jae Hee Lee, Stefan Wermter (2024). Read Between the Layers: Leveraging Multi-Layer Representations for Rehearsal-Free Continual Learning with Pre-Trained Models. In _Transactions on Machine Learning Research (TMLR)_.

[Paper](https://openreview.net/pdf?id=ZTcxp9xYr2) • [OpenReview](https://openreview.net/forum?id=ZTcxp9xYr2) • [ArXiV](https://arxiv.org/abs/2312.08888)

------------------------

## Overview
![LayUP Approach](
    img/layup_overview.png
)


------------------------


## To reproduce the results of the paper, please follow the instructions below.

Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

Run the following command to train and evaluate the model:
```bash
python main.py --dataset [dataset id] --finetune_method [fine-tuning method] --backbone [pre-trained model]
```

For more information about additional parameters that can be set (e.g., number of tasks _T_ or maximum layer depth _k_):
```bash
python main.py --help
```

### Datasets

| Dataset ID | Full Name | Download Link | Automatic Download |
| --- | --- | --- | --- |
| cifar100 | CIFAR-100 | [Link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | ✅ |
| imagenetr | ImageNet-R | [Link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR) | ❌ |
| imageneta | ImageNet-A | [Link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p) | ❌ |
| vtab | VTAB | [Link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_) | ❌ |
| cars | Cars-196 | [Link](https://drive.google.com/file/d/1MbAlm4ciYNtWhMVL8K8_uxIcFtes2_jI) | ❌ |
| cub | CUB200 | [Link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb) | ❌ |
| omnibenchmark | OmniBenchmark | [Link](https://drive.google.com/file/d/1GozYc4T5s3MkpEtYRoRhW92T-0sW4pFV) | ❌ |
| dil_imagenet_r | DIL ImageNet-R | cf. ImageNet-R | ❌ |
| cddb | CDDB-Hard | [Link](https://coral79.github.io/CDDB_web/) | ❌ |
| limited_domainnet | DomainNet | [Link](http://ai.bu.edu/M3SDA/) | ✅ |

### Parameter-Efficient Fine-Tuning Methods

| Fine-tuning Method | Reference |
| --- | --- |
| none | No finetuning |
| adapter | [AdaptFormer Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html) | 
| ssf | [SSF Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html) |
| vpt | [VPT Paper](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_41) |

### Backbones

| Pre-trained Model | Full Name | Library | Automatic Download |
| --- | --- | --- | --- |
| vit_base_patch16_224 | ViT-B/16-IN1K | [timm](https://github.com/huggingface/pytorch-image-models) | ✅ |
| vit_base_patch16_224_in21k | ViT-B/16-IN21K | [timm](https://github.com/huggingface/pytorch-image-models) | ✅ |

------------------------

## Cite our work

If you find our work on continual learning useful, please cite our work:
```bash
@article{
ahrens_2024_layup,
title={Read Between the Layers: Leveraging Multi-Layer Representations for Rehearsal-Free Continual Learning with Pre-Trained Models},
author={Kyra Ahrens and Hans Hergen Lehmann and Jae Hee Lee and Stefan Wermter},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=ZTcxp9xYr2}
}
```

**Please feel free to contact us if you have any questions! You find the authors' email addresses on the paper header.** 
