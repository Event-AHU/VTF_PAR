import torch.nn as nn
import torch
from CLIP.clip import clip
from models.vit import *
from CLIP.CoOp import *
device = "cuda" if torch.cuda.is_available() else "cpu"
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num,attr_words, dim=768, pretrain_path='/amax/DATA/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(512, dim)
        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)
        self.text = clip.tokenize(attr_words).to(device)

    def forward(self,videos,ViT_model):
        ViT_features=[]
        if len(videos.size())<5 :
            videos.unsqueeze(1)
        batch_size, num_frames, channels, height, width = videos.size()
        imgs=videos.view(-1, channels, height, width)
        #imgs=videos[:,0,:,:,:]
        #CLIP 提取视频帧特征
        for img in imgs:
            img=img.unsqueeze(0)
            ViT_features.append(ViT_model.encode_image(img).squeeze(0))
            #图像特征
        ViT_image_features=torch.stack(ViT_features).to(device).float()
        
        _,token_num,visual_dim=ViT_image_features.size()
        ViT_image_features=ViT_image_features.view(batch_size,num_frames,token_num,visual_dim)
        
        ViT_image_features=self.visual_embed(torch.mean(ViT_image_features,dim=1))
        text_features = ViT_model.encode_text(self.text).to(device).float()
        textual_features = self.word_embed(text_features).expand(ViT_image_features.shape[0], text_features.shape[0],768)    

        x = torch.cat([textual_features,ViT_image_features], dim=1)

        for b_c,blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)
        return logits
    
