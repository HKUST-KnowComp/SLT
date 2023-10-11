# KnowComp Submission for WMT23 Sign Language Translation Task
This is the official repository for the workshop paper in WMT-SLT23: KnowComp Submission for WMT23 Sign Language Translation Task. We endeavor to train an embedding alignment block to align the visual modality and text modality in order to generate more reasonable natural language.
![Model demonstration](https://github.com/HKUST-KnowComp/SLT/blob/main/model_figure.pdf)
## Weight downloading
To download our model weights and tokenizer, please download it from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bxuan_connect_ust_hk/EkcGHlIfS5hLvkIW1JgA_QsB542tOQxIHVppw_tXamN4IA?e=3ohsw6). If you want to download the training data, please refer to official websit of WMT-SLT23 and apply for the license.

## Required Packages
Required packages are listed in `requirements.txt`. Install them by running:

```bash
pip install -r requirements.txt
```

## Citing this work
Please use the bibtex below to cite our paper:
```bibtex
@inproceedings{WMT23SLT_knowcomp,
  author       =  {Baixuan Xu and
                   Haochen Shi and
                   Tianshi Zheng and
                   Qing Zong and
                   Weiqi Wang and
                   Zhaowei Wang and
                   Yangqiu Song},
  title        = {KnowComp Submission for WMT23 Sign Language Translation Task},
  month = {dec},
  year         = {2023},
  booktitle = {Proceedings of the Eighth Conference on Machine Translation, WMT 2023}
}
```