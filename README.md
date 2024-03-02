

# VTF_PAR
> **Learning CLIP Guided Visual-Text Fusion Transformer for Video-based Pedestrian Attribute Recognition**, Jun Zhu†, Jiandong Jin†, Zihan Yang, Xiaohao Wu, Xiao Wang († denotes equal contribution), CVPR-2023 Workshop@NFVLR, pp. 2625-2628. 2023. (New Frontiers in Visual Language Reasoning: Compositionality, Prompts and Causality),  
[[arXiv](https://arxiv.org/abs/2304.10091)]
[[CVF](https://openaccess.thecvf.com/content/CVPR2023W/NFVLR/papers/Zhu_Learning_CLIP_Guided_Visual-Text_Fusion_Transformer_for_Video-Based_Pedestrian_Attribute_CVPRW_2023_paper.pdf)] 
[[Workshop](https://nfvlr-workshop.github.io/)] 


## Abstract 
Existing pedestrian attribute recognition (PAR) algorithms are mainly developed based on a static image. However, the performance is not reliable for images with challenging factors, such as heavy occlusion, motion blur, etc. In this work, we propose to understand human attributes using video frames that can make full use of temporal information. Specifically, we formulate the video-based PAR as a vision-language fusion problem and adopt pre-trained big models CLIP to extract the feature embeddings of given video frames. To better utilize the semantic information, we take the attribute list as another input and transform the attribute words/phrase into the corresponding sentence via split, expand, and prompt. Then, the text encoder of CLIP is utilized for language embedding. The averaged visual tokens and text tokens are concatenated and fed into a fusion Transformer for multi-modal interactive learning. The enhanced tokens will be fed into a classification head for pedestrian attribute prediction. Extensive experiments on a large-scale video-based PAR dataset fully validated the effectiveness of our proposed framework. 

<img src="https://github.com/Event-AHU/VTF_PAR/blob/main/figures/frameworkV4.jpg" width="800">
### Requirements
we use single RTX3090 20G GPU for training and evaluation. 
Basic Environment
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
```
Installation
```
pip install -r requirements.txt
```
## Datasets and Pre-trained Models 

**Download from BaiduYun:**

* **MARS Dataset**:
```
链接：https://pan.baidu.com/s/16Krv3AAlBhB9JPa1EKDbLw 提取码：zi08
```

* **Pre-trained Models (VTF-Pretrain.pth)**:
```
链接：https://pan.baidu.com/s/150t_zCW35YQHViKxsRIVzQ  提取码：glbd
```

**Download from DropBox:**
```
https://www.dropbox.com/scl/fo/h70nbcuj4gsmi4txhq1i0/h?rlkey=rwn1gbqbjpak6d7zhp46o3rnb&dl=0
``` 




## Training and Testing 
Use the following code to learn a model for MARS Dataset:

Training
```
python ./dataset/preprocess/mars.py
python train.py MARS
```
Testing
```
python eval.py MARS
```
## :page_with_curl: BibTex: 
If you find this work useful for your research, please cite the following papers: 

```bibtex
@inproceedings{zhu2023learning,
  title={Learning CLIP Guided Visual-Text Fusion Transformer for Video-based Pedestrian Attribute Recognition},
  author={Zhu, Jun and Jin, Jiandong and Yang, Zihan and Wu, Xiaohao and Wang, Xiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2625--2628},
  year={2023}
}
```

If you have any questions about this work, please submit an issue or contact me via **Email**: wangxiaocvpr@foxmail.com or xiaowang@ahu.edu.cn. Thanks for your attention! 




