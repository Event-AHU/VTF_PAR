import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn,optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler

from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter
set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"
ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/amax/DATA/jinjiandong/model') 
def main(args):
    log_dir = os.path.join('logs', args.dataset)
    tb_writer = SummaryWriter('/amax/DATA/jinjiandong/CaptionCLIP-ViT-B/tensorboardX/exp')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)

    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args) 

    train_set = MultiModalAttrDataset(args=args, split=args.train_split , transform=train_tsfm) 

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=8,     
        pin_memory=True,  
    )
    
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split , transform=valid_tsfm) 

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    labels = train_set.label
    sample_weight = labels.mean(0)
    model = TransformerClassifier(train_set.attr_num,attr_words=train_set.attributes,length=args.length)
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint=torch.load('/amax/DATA/jinjiandong/PromptHAR-Finetuning_zj/clip.pth')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    criterion = CEL_Sigmoid(sample_weight,attr_idx=train_set.attr_num)
    lr = args.lr
    epoch_num = args.epoch
    start_epoch=1
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)




    best_metric, epoch = trainer(args=args,
                                 epoch=epoch_num,
                                 model=model,
                                 ViT_model=ViT_model,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir,
                                 tb_writer=tb_writer,
                                 start_epoch=start_epoch)
    
def trainer(args,epoch, model,ViT_model, valid_loader, criterion):
    valid_loss, valid_gt, valid_probs,valid_name = valid_trainer(
        epoch=epoch,
        model=model,
        ViT_model=ViT_model,
        valid_loader=valid_loader,
        criterion=criterion,
    )
    if args.dataset =='MARS' : 
        #MARS
        index_list=[0,1,2,3,4,5,6,7,8,9,15,20,29,39,43]
        group="top length, bottom length, shoulder bag, backpack, hat, hand bag, hair, gender, bottom type, pose, motion, top color, bottom color, age"
    else:
        #DUKE
        index_list=[0,1,2,3,4,5,6,7,8,14,19,28,36]
        group="backpack, shoulder bag, hand bag, boots, gender, hat, shoes, top length, pose, motion, top color, bottom color"
    group_ma=[]
    group_f1=[]
    group_acc=[]
    group_prec=[]
    group_recall=[]
    for idx in range(len(index_list)-1):
        if index_list[idx+1]-index_list[idx] >1 :
            result=get_pedestrian_metrics(valid_gt[:,index_list[idx]:index_list[idx+1]], valid_probs[:,index_list[idx]:index_list[idx+1]])
        else  :
            result=get_signle_metrics(valid_gt[:,index_list[idx]], valid_probs[:,index_list[idx]])
    group_ma.append(result.ma)
    group_f1.append(result.instance_f1) 
    group_acc.append(result.instance_acc)  
    group_prec.append(result.instance_prec)
    group_recall.append(result.instance_recall)   
    group_all= [group_ma,group_f1,group_acc,group_prec,group_recall]
    average_ma = np.mean(group_ma)
    average_instance_f1 = np.mean(group_f1)
    average_acc = np.mean(group_acc)
    average_prec = np.mean(group_prec)    
    average_recall = np.mean(group_recall)
    average_all=[average_ma,average_instance_f1,average_acc,average_prec,average_recall]    
    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

    print(f'{time_str()}Evaluation on test set, valid_loss:{valid_loss:.4f}\n',
        f"ma :{group} \n",','.join(str(elem)[:6] for elem in group_ma),'\n',
        f"Acc :",','.join(str(elem)[:6] for elem in group_acc),'\n',
        f"Prec :",','.join(str(elem)[:6] for elem in group_prec),'\n',
        f"Recall :",','.join(str(elem)[:6] for elem in group_recall),'\n',
        f"F1 :{group}  \n",','.join(str(elem)[:6] for elem in group_f1),'\n',
        'average_ma: {:.4f},  average_acc: {:.4f},average_prec: {:.4f},average_recall: {:.4f},average_f1: {:.4f}'.format(average_ma,average_acc , average_prec, average_recall, average_instance_f1)                 
        )
    print('-' * 60)     

    
        

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()