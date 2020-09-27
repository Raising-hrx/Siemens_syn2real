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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_dir', default='output/debug', type=str,
                        help='directory saving model and gshape2class.json')
    parser.add_argument('--model', default='debug.pth', type=str,
                        help='name of model used for inference')
    
    parser.add_argument('--run_on_dataset', action='store_true', default=False,
                        help='if true: inference on dataset; else inference on single image')
    parser.add_argument('--test_path', default='', type=str,
                        help='directory to test dataset or image') 

    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    
    parser.add_argument('--img_size', default=360, type=int,
                        help='image size for training and testing')
    parser.add_argument('--feature_len', default=2048, type=int,
                        help='the length of the finnal feature before fc')
    parser.add_argument('--dropout_rate', default=0.0, type=float,
                        help='dropout rate for dropout layer, 0 for not using')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

from train import Multitask_Model


def infer_dataset(args):
    # build test transforms, datasets and dataloader
    transforms_test = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.img_size),       # resize to 360*? or ?*360
                    transforms.CenterCrop(args.img_size),   # crop center 360*360
                    transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    ])
    dataset_test = TestDateset(args.test_path,transforms_test,args)
    
    
    dataloader_test = DataLoader(dataset_test,
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                drop_last=False,)
    
    gshape2class = dataset_test.gshape2class
    shape_total_cls = len(gshape2class)
    
    model = Multitask_Model(shape_total_cls, args)
    
    # load trained parameters
    path = osp.join(args.output_dir, args.model)
    model_dict_load = torch.load(path)
    model_dict = model.state_dict()
    model_dict.update(model_dict_load)

    model.load_state_dict(model_dict)
    
    model.to(device)
    
    # start inference
    model.eval()

    iter_loader_test = iter(dataloader_test)
    correct_count = 0
    for input_data in tqdm(iter_loader_test):    # input_data = next(iter_loader_test)
        img,labels = input_data
        label_cls = labels[:,0].long()

        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            shape_pred, _, _ = output  # softmax can be ignored ?
            
        cls_pred = shape_pred.argmax(dim=1)
        cls_pred = cls_pred.cpu()

        for i in range(len(cls_pred)):
            cls_pred[i] = gshape2class[str(int(cls_pred[i]))]

        correct_count += cls_pred.eq(label_cls).sum()

    acc = correct_count.float() / len(dataset_test)
    
    print("accuracy in test dataset is : {:.2f}%".format(100*float(acc)))
    
    return
                    

def infer_one_image(args):
    
    # build test transforms
    transforms_test = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.img_size),       # resize to 360*? or ?*360
                    transforms.CenterCrop(args.img_size),   # crop center 360*360
                    transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    ])
    # get global shape to class dict and class name
    with open(os.path.join(args.output_dir,'gshape2class.json'),'r') as f:
        gshape2class = json.load(f)

    class_name = ['F58001104805002120001',
                 'F58001104827202120004',
                 'F58001104821701010008',
                 'F58001104949503200002',
                 'F58000104805003240003',
                 'F58001104805003210004']
    
    shape_total_cls = len(gshape2class)
    
    # bulid model
    model = Multitask_Model(shape_total_cls, args)
    
    # load trained parameters
    path = osp.join(args.output_dir, args.model)
    model_dict_load = torch.load(path)
    model_dict = model.state_dict()
    model_dict.update(model_dict_load)

    model.load_state_dict(model_dict) 
    
    img = cv2.cvtColor(cv2.imread(args.test_path),cv2.COLOR_BGR2RGB)
    img = transforms_test(img)
    
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        shape_pred, _, _ = output 

    cls_pred = shape_pred.argmax(dim=1)
#     cls_pred = cls_pred.cpu()
    cls_pred = cls_pred[0]
    cls_pred = gshape2class[str(int(cls_pred))]
    
    print("This image belongs to the {:d}-th calss({:s})".format(cls_pred,class_name[cls_pred]))
    
    
if __name__ == "__main__":
    args = parse_args()
    
    
    # check device
    device = torch.device('cuda')
    
    print(args.run_on_dataset)
    # do main
    if args.run_on_dataset:
        infer_dataset(args)
    else:
        infer_one_image(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    