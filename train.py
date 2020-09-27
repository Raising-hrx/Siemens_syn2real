#!/usr/bin/env python3

import sys
import os
import os.path as osp

import pickle
import argparse
import json
import random
from datetime import datetime
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
import torch.nn.functional as F

from data.dataset import TrainDateset,TestDateset

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--comment', default='debug', type=str,
                        help='comment for this experiment')
    parser.add_argument('--output_dir', default='output/debug', type=str,
                        help='directory to save results and logs')
    
    parser.add_argument('--train_path', default='', type=str,
                        help='directory to train dataset') 
    parser.add_argument('--test_path', default='', type=str,
                        help='directory to test dataset') 

    parser.add_argument('--base_lr', default=0.0001, type=float,
                        help='base learning rate for optimizer')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')

    parser.add_argument('--max_epoch',default=120, type=int,
                        help='the max epoch number for training') # 最大训练轮数
    parser.add_argument('--val_epoch', default=1, type=int,
                        help='validate model each x epoch on test data') # 每x轮在测试集上进行一次验证 
    parser.add_argument('--val_epoch_train', default=5, type=int,
                        help='validate model each x epoch on training data') # 每x轮在训练集上进行一次验证
    parser.add_argument('--save_epoch', default=5, type=int,
                        help='save model each x epoch') # 每x轮保存一次模型
    parser.add_argument('--save_start_epoch', default=50, type=int,
                        help='start to save model after this epoch') # 在第x轮之后才开始保存模型
    
    
    parser.add_argument('--img_size', default=360, type=int,
                        help='image size for training and testing')
    parser.add_argument('--feature_len', default=2048, type=int,
                        help='the length of the finnal feature before fc')
    parser.add_argument('--dropout_rate', default=0.0, type=float,
                        help='dropout rate for dropout layer, 0 for not using')
    
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

class Multitask_Model(nn.Module):
    def __init__(self,shape_total_cls, args=None):
        super(Multitask_Model, self).__init__()

        self.dropout_rate = args.dropout_rate
        
        self.feature_extracter = models.resnet50(pretrained=True) # 预训练resnet50模型
        self.feature_extracter.fc = nn.Linear(in_features=2048,out_features=args.feature_len,bias=False)

        self.shape_fc = nn.Linear(in_features=args.feature_len,out_features=shape_total_cls,bias=True)
        self.angle_fc = nn.Linear(in_features=args.feature_len,out_features=1,bias=True)
        self.color_fc = nn.Linear(in_features=args.feature_len,out_features=3,bias=True)
        self.dropout = nn.Dropout(args.dropout_rate)
            
        self._fix_feature_layer(fixed_layer = 3)

    def _fix_feature_layer(self,fixed_layer = 3):
        for name,p in self.feature_extracter.named_parameters():
            if name.startswith("layer"+str(fixed_layer+1)) or name.startswith("fc"):
                break
            p.requires_grad = False
        #     print(name)

        print("train the following parameters")
        for name,p in self.named_parameters():     
            if p.requires_grad:
                print(name)

    def forward(self,img):
        features = F.relu( self.feature_extracter(img) )
    
        # use dropout
        if self.dropout_rate > 0.0:  
            features = self.dropout(features)
        
        shape_pred = self.shape_fc(features)
        angle_pred = self.angle_fc(features)
        color_pred = self.color_fc(features)

        # use sigmoid for reg [0~1] task
        angle_pred = torch.sigmoid(angle_pred)
        color_pred = torch.sigmoid(color_pred) 
        
        return shape_pred, angle_pred, color_pred

def start_train(args):
    # 收集训练数据
    transforms_train = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.img_size),       # 短边缩放  （输入图像大部分为720*1280*3）
                    transforms.CenterCrop(args.img_size),   # 中心截取
                    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), 
                                            shear=None, resample=False, fillcolor=0), # 随机平移与尺度
                    transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    ])

    dataset_train = TrainDateset(args.train_path,transforms_train,args)

    # 收集测试数据
    transforms_test = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.img_size),       # 短边缩放  （输入图像大部分为720*1280*3）
                    transforms.CenterCrop(args.img_size),   # 中心截取
                    transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    ])
    dataset_test = TestDateset(args.test_path,transforms_test,args)


    dataloader_train = DataLoader(dataset_train,
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=4, 
                                    drop_last=False,)

    dataloader_test = DataLoader(dataset_test,
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=4, 
                                    drop_last=False,)


    gshape2class = dataset_train.gshape2class # 形状到类别的词典
    shape_total_cls = len(gshape2class)
    angle_totcl_cls = 360 // dataset_train.para_angle_cls


    model = Multitask_Model(shape_total_cls, args)

    params_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params_train, lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = torch.optim.SGD(params_train, lr=0.001)


    loss_CE = torch.nn.CrossEntropyLoss()
    loss_L1 = torch.nn.L1Loss()


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    #     model = nn.DataParallel(model,device_ids=[0,1,2,3])
#     device = torch.device("cuda:0")
    model.to(device)


    best_test_acc = 0
    for epoch in range(args.max_epoch):

        # train epoch
        loss_smooth = 0
        iter_loader_train = iter(dataloader_train)
        for input_data in tqdm(iter_loader_train):
        # input_data = next(iter_loader_train)
            model.train()

            img,labels = input_data
            img,labels = img.to(device),labels.to(device)

            label_angle = labels[:,2].float() / 360.0
            label_shape = labels[:,3].long()
            label_rgb = labels[:,4:].float()

            output = model(img)
            shape_pred, angle_pred, color_pred = output
 
            loss_shape = loss_CE(shape_pred,label_shape) * 1 

            loss_angle = loss_L1(angle_pred,label_angle) * 50

            loss_color = loss_L1(color_pred,label_rgb) * 100

            loss = loss_shape + loss_angle + loss_color

            with open(osp.join(args.output_dir,"over_record.txt"),'a+') as f:
                f.write("training epoch {} \t loss_shape {:.4f} \t loss_angle {:.4f} \t loss_color {:.4f} \t \n".format(epoch+1,loss_shape,loss_angle,loss_color))


            loss_smooth += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with open(osp.join(args.output_dir,"record.txt"),'a+') as f:
            f.write("training epoch {} \t loss {:.4f} \n".format(epoch+1,loss_smooth/len(dataloader_train)))


        if not (epoch+1)%args.val_epoch:
            # test epoch
            model.eval()

            iter_loader_test = iter(dataloader_test)
            correct_count = 0
            for input_data in tqdm(iter_loader_test):
            # input_data = next(iter_loader_test)
                img,labels = input_data
                label_cls = labels[:,0].long()

                img = img.to(device)

                with torch.no_grad():
                    output = model(img)
                    shape_pred, _, _ = output

                cls_pred = shape_pred.argmax(dim=1)
                cls_pred = cls_pred.cpu()

                for i in range(len(cls_pred)):
                    cls_pred[i] = gshape2class[int(cls_pred[i])]

                correct_count += cls_pred.eq(label_cls).sum()

            acc = correct_count.float() / len(dataset_test)

            with open(osp.join(args.output_dir,"record.txt"),'a+') as f:
                f.write("test epoch {} \t test dataset accuracy {:.4f} \n".format(epoch+1,acc))

            if acc >= best_test_acc:
                best_test_acc = acc
                filepath = osp.join(args.output_dir,"model_dict_best.pth")
                if os.path.exists(filepath):
                    os.remove(filepath)
                if isinstance(model,nn.DataParallel):
                    torch.save(model.module.state_dict(),filepath) # .module for DataParallel instance
                else:
                    torch.save(model.state_dict(),filepath)

        if not (epoch+1)%args.val_epoch_train:
            # test epoch
            model.eval()

            iter_loader_train = iter(dataloader_train)
            correct_count = 0
            for input_data in tqdm(iter_loader_train):
                img,labels = input_data
                label_cls = labels[:,0].long()

                img = img.to(device)

                with torch.no_grad():
                    output = model(img)
                    shape_pred, _, _ = output

                cls_pred = shape_pred.argmax(dim=1)
                cls_pred = cls_pred.cpu()

                for i in range(len(cls_pred)):
                    cls_pred[i] = gshape2class[int(cls_pred[i])]

                correct_count += cls_pred.eq(label_cls).sum()

            acc = correct_count.float() / len(dataset_train)

            with open(osp.join(args.output_dir,"record.txt"),'a+') as f:
                f.write("test epoch {} \t train dataset accuracy {:.4f} \n".format(epoch+1,acc))

        if (not (epoch+1)%args.save_epoch) and (epoch > args.save_start_epoch):
            filepath = osp.join(args.output_dir,"model_dict_epoch{}.pth".format(int((epoch+1))))
            
            if isinstance(model,nn.DataParallel):
                torch.save(model.module.state_dict(),filepath) # .module for DataParallel instance
            else:
                torch.save(model.state_dict(),filepath)

if __name__=='__main__':
    
    args = parse_args()
    
    # set and make output dir
    args.output_dir = osp.join(args.output_dir, args.comment+'-'+datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(args.output_dir,exist_ok=True)
    
    # dump config.json
    with open(osp.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print(args)
    
    
    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.output_dir))
    os.system('cp -r *.py {}'.format(args.output_dir)) ## TODO
    
    # check device
    device = torch.device('cuda')
    
    # set random seed before init model
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---do main
    start_train(args)
