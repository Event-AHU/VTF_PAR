import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
import time
from torch import nn,optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from CLIP.clip import clip
from CLIP.clip.model import *
import torchvision.transforms as T
from PIL import Image
set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"
ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/amax/DATA/jinjiandong/model') 
attr_words = [
    'top short', #top length 0 
    'bottom short', #bottom length 1
    'shoulder bag','backpack',#shoulder bag #backpack 2 3
    'hat', 'hand bag', 'long hair', 'female',# hat/hand bag/hair/gender 4 5 6 7
    'bottom skirt', #bottom type 8
    'frontal', 'lateral-frontal', 'lateral', 'lateral-back', 'back', 'pose varies',#pose[9:15]
    'walking', 'running','riding', 'staying', 'motion varies',#motion[15:20]
    'top black', 'top purple', 'top green', 'top blue','top gray', 'top white', 'top yellow', 'top red', 'top complex',#top color [20 :29]
    'bottom white','bottom purple', 'bottom black', 'bottom green', 'bottom gray', 'bottom pink', 'bottom yellow','bottom blue', 'bottom brown', 'bottom complex',#bottom color[29:39]
    'young', 'teenager', 'adult', 'old'#age[39:43]
]

#img or video path
img_path = ''
checkpoint_path = ''
parser = argument_parser()
args = parser.parse_args()
normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    normalize
])

imgs=[]
for i in  os.listdir(img_path):
    pil=Image.open(i)
    imgs.append(transform(pil))
img_tensor=torch.stack(imgs).to(device)
model = TransformerClassifier(attr_num=43,attr_words=attr_words,length=args.length)
model = model.to(device)
ViT_model = ViT_model.to(device)
checkpoint=torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'],strict=False)
logits = model(imgs,ViT_model=ViT_model)
pred_result = torch.sigmoid(logits).detach().cpu().numpy()>0.45

