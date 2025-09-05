# Large Margin Representation Learning for Robust Cross-lingual Named Entity Recognition

This is the implementation of our ACL 2025 paper [MARAL](https://aclanthology.org/2025.acl-long.215/).

**Title:** Large Margin Representation Learning for Robust Cross-lingual Named Entity Recognition  
**Authors:** Guangcheng Zhu, Ruixuan Xiao, Haobo Wang, Zhen Zhu, Gengyu Lyu, Junbo Zhao  
**Affiliations:** Zhejiang University, Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security, Zhejiang Sci-tech University, Beijing University of Technology

## Overview

In this work, we propose a novel framework MARAL for cross-lingual NER. We first identify that existing methods struggle to achieve the optimal features with maximum margins due to two key issues, i.e., the inherent distribution skewness and pseudo-label bias. To address this, we derive an adaptively reweighted contrastive objective to maximize margins by explicit distribution modeling and performing adaptive inter-class separation. To further address unreliable pseudo-labels, we incorporate a progressive adaptation strategy, which first selects reliable samples as anchors and refines the remaining unreliable ones.

An overview of our proposed MARAL can be seen as follows:
![MARAL Overview](./pipeline.png)

## Running

### Dependencies
To install requirements:
```bash
pip install requirements.txt
```

### Data Preparation
All the datasets we used are publicly available datasets. For convenience, it is recommened to put the data for TRAILER under the `data` folder with the following structure:
  > data
  >
  > ├── dataset_language
  > 
  > │   ├── spanner.train
  >
  > │   ├── spanner.dev
  >
  > │   ├── spanner.unlabel
  >
  > │   └── spanner.test

The data needs to follow the span-based format and we have provided the processed German dataset from CoNLL as an example in `data/conll03_de`. You can obtain the entire processed datasets from [Google Drive](https://drive.google.com/drive/folders/1Opua13lXGMVFckQ6Z_lndMX8EYhc0fbY?usp=sharing).

### Pretrained Models
The multilingual pretrained language model `xlm-roberta-large` is adopted as the backbone following previous protocols. The model can be downloaded from [Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-large) and placed under the `pretrained` folder.

### Training Scripts
To generate pseudo labels for the target-language data, run the following command:
```bash
bash generate_pseudo_labels.sh de conll03
```

To train the target student model, run the following command:
```bash
bash train_tgt_model.sh de conll03
```

**Note:** For different languages, you may adjust the hyperparameters `tau` and `ul_tau` in `train_tgt_model.py` to control the strength of loss reweighting, in an effort to achieve better performance.

## Acknowledgements
Our code framework refers to [ContProto](https://github.com/DAMO-NLP-SG/ContProto) and [GLoDe](https://github.com/Ding-ZJ/GLoDe), many thanks!

## Citation
If you find this code useful, please consider citing our paper:
```cite
@inproceedings{zhu2025large,
  title={Large Margin Representation Learning for Robust Cross-lingual Named Entity Recognition},
  author={Zhu, Guangcheng and Xiao, Ruixuan and Wang, Haobo and Zhu, Zhen and Lyu, Gengyu and Zhao, Junbo},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={4270--4291},
  year={2025}
}
```

## Contact
If you have any questions, please feel free to contact the authors. Guangcheng Zhu: zhuguangcheng@zju.edu.cn.