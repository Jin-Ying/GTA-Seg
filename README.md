# GTA-Seg
Code release for [Semi-Supervised Semantic Segmentation via Gentle Teaching Assistant], NeurIPS 2022.

## Installation

```bash
cd GTA-Seg
conda create -n gta python=3.6.9
conda activate gta
pip install -r requirements.txt
pip install pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
### Prepare Data
<details>
  <summary>For PASCAL VOC 2012</summary>

Download "VOCtrainval_11-May-2012.tar" from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar.

And unzip the files to folder ```data``` and make the dictionary structures as follows:

```angular2html
data/VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
├── SegmentationClassAug
└── SegmentationObject
```
</details>

### Prepare Pretrained Backbone

Before training, please download ResNet101 pretrained on ImageNet-1K from one of the following:
  - [Google Drive](https://drive.google.com/file/d/1nzSX8bX3zoRREn6WnoEeAPbKYPPOa-3Y/view?usp=sharing)
  - [Baidu Drive](https://pan.baidu.com/s/1FDQGlhjzQENfPp4HTYfbeA) Fetch Code: 3p9h

After that, modify ```model_urls``` in ```models/resnet.py``` to ```</path/to/resnet101.pth>```
### Train a Semi-Supervised Model

We can train a model on PASCAL VOC 2012 with ```183``` labeled data for supervision by:
```bash
cd experiments/pascal/183/ours
# use slurm
sh slurm_train.sh <num_gpu> <port> <partition>
# or use torch.distributed.launch
# sh train.sh <num_gpu> <port>
```

## Acknowledgement

We reproduce our work based on **U2PL**.
- U2PL: https://github.com/Haochen-Wang409/U2PL

Sincere gratitude to their work.

## Citation
```bibtex
@inproceedings{jin2022semi,
    title={Semi-Supervised Semantic Segmentation via Gentle Teaching Assistant},
    author={Jin, Ying and Wang, Jiaqi and Lin, Dahua},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```

## Contact

- Ying Jin, sherryying003@gmail.com