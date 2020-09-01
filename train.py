import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import sys
import pickle
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
import torch.nn.functional as F

from data.dataset_v1 import TrainDateset,TestDateset

# 训练超参数
BASE_LR = 0.0005 # 0.001 # 优化器学习率
BATCH_SIZE = 16*6  # batch大小

MAX_EPOCH = 120 # 最大训练轮数
VAL_EPOCH = 1  # 每x轮在测试集上进行一次验证 
VAL_EPOCH_TRAIN = 5 # 每x轮在训练集上进行一次验证
SAVE_PRE_EPOCH = 5 # 每x轮保存一次模型
SAVE_START_EPOCH = 50 # 在第x轮之后才开始保存模型

# 保存模型与日志的路径设置
comment = "multitask_0831_seed2020_t"
record_fold = "Outputs/" + comment + "/train_stage1"
os.makedirs(record_fold,exist_ok=True)
record_path = osp.join(record_fold,"record.txt")

# 训练集与测试集路径设置
train_path = "/home/hongruixin/Siemens-project/data/data_v1/" 
test_path =  "/home/hongruixin/Siemens-project/data/data_v1/real_image_wo_repeat/"

# 设置随机种子
random_seed = 2020
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.backends.cudnn.deterministic = True


# 收集训练数据
transforms_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(360),       # 短边缩放  （输入图像大部分为720*1280*3）
                transforms.CenterCrop(360),   # 中心截取
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), shear=None, resample=False, fillcolor=0), # 随机平移与尺度
                transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])

dataset_train = TrainDateset(train_path,transforms_train)

# 收集测试数据
transforms_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(360),       # 短边缩放  （输入图像大部分为720*1280*3）
                transforms.CenterCrop(360),   # 中心截取
                transforms.ToTensor(),        # PIL 0-255  -->  tensor 0-1.0
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])
dataset_test = TestDateset(test_path,transforms_test)


dataloader_train = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=0, 
                                drop_last=False,)

dataloader_test = DataLoader(dataset_test,
                                batch_size=BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=0, 
                                drop_last=False,)


gshape2class = dataset_train.gshape2class # 形状到类别的词典
shape_total_cls = len(gshape2class)
angle_totcl_cls = 360 // dataset_train.para_angle_cls


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.feature_extracter = models.resnet50(pretrained=True) # 预训练resnet50模型
        self.feature_extracter.fc = nn.Linear(in_features=2048,out_features=2048,bias=True)

        self.shape_cls_fc = nn.Linear(in_features=2048,out_features=shape_total_cls,bias=True)
        self.angle_cls_fc = nn.Linear(in_features=2048,out_features=angle_totcl_cls,bias=True)
        self.color_reg_fc = nn.Linear(in_features=2048,out_features=3,bias=True)
        
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
        
        shape_pred = self.shape_cls_fc(features)
        angle_pred = self.angle_cls_fc(features)
        color_pred = self.color_reg_fc(features)
        
        return shape_pred, angle_pred, color_pred
#         return shape_pred, None, None

model = MyModel()

params_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params_train, lr=BASE_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = torch.optim.SGD(params_train, lr=0.001)


loss_fn = torch.nn.CrossEntropyLoss()
loss_fn_L1 = torch.nn.L1Loss()


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
#     model = nn.DataParallel(model,device_ids=[0,1,2,3])
device = torch.device("cuda:0")
model.to(device)


best_test_acc = 0
for epoch in range(MAX_EPOCH):
    
    # train epoch
    loss_smooth = 0
    iter_loader_train = iter(dataloader_train)
    for input_data in tqdm(iter_loader_train):
    # input_data = next(iter_loader_train)
        model.train()
        
        img,labels = input_data
        img,labels = img.to(device),labels.to(device)
        
        label_angle = labels[:,2].long()
        label_shape = labels[:,3].long()
        label_rgb = labels[:,4:].float()
        
        output = model(img)
        shape_pred, angle_pred, color_pred = output
        
        shape_pred = torch.sigmoid(shape_pred) 
        loss_shape = loss_fn(shape_pred,label_shape)

        angle_pred = torch.sigmoid(angle_pred)
        loss_angle = loss_fn(angle_pred,label_angle)
        
        loss_color = loss_fn_L1(color_pred,label_rgb)
        
        loss = loss_shape + loss_angle + loss_color
        
#         with open(osp.join(record_fold,"over_record.txt"),'a+') as f:
#             f.write("training epoch {} \t loss_shape {:.4f} \t loss_angle {:.4f} \t loss_color {:.4f} \t \n".format(epoch+1,
#                                                                                                                     loss_shape,
#                                                                                                                     loss_angle,
#                                                                                                                     loss_color))

#         print(loss_shape,loss_angle,loss_color)
#         loss = loss_shape

        loss_smooth += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with open(record_path,'a+') as f:
        f.write("training epoch {} \t loss {:.4f} \n".format(epoch+1,loss_smooth/len(dataloader_train)))
        
        
    if not (epoch+1)%VAL_EPOCH:
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
                
                shape_pred = torch.sigmoid(shape_pred)

            cls_pred = shape_pred.argmax(dim=1)
            cls_pred = cls_pred.cpu()
            
            for i in range(len(cls_pred)):
                cls_pred[i] = gshape2class[int(cls_pred[i])]
            
            correct_count += cls_pred.eq(label_cls).sum()

        acc = correct_count.float() / len(dataset_test)
        
        with open(record_path,'a+') as f:
            f.write("test epoch {} \t test dataset accuracy {:.4f} \n".format(epoch+1,acc))
            
        if acc >= best_test_acc:
            best_test_acc = acc
            filepath = osp.join(record_fold,"model_dict_best.pth")
            if os.path.exists(filepath):
                os.remove(filepath)
            torch.save(model.module.state_dict(),filepath) # .module for DataParallel instance
        
    if not (epoch+1)%VAL_EPOCH_TRAIN:
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
                
                shape_pred = torch.sigmoid(shape_pred)

            cls_pred = shape_pred.argmax(dim=1)
            cls_pred = cls_pred.cpu()
            
            for i in range(len(cls_pred)):
                cls_pred[i] = gshape2class[int(cls_pred[i])]
            
            correct_count += cls_pred.eq(label_cls).sum()

        acc = correct_count.float() / len(dataset_train)
        
        with open(record_path,'a+') as f:
            f.write("test epoch {} \t train dataset accuracy {:.4f} \n".format(epoch+1,acc))
            
    if (not (epoch+1)%SAVE_PRE_EPOCH) and (epoch > SAVE_START_EPOCH):
        filepath = osp.join(record_fold,"model_dict_epoch{}.pth".format(int((epoch+1))))
        torch.save(model.module.state_dict(),filepath) # .module for DataParallel instance