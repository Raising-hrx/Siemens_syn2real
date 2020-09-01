import os
import sys
import pickle
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

class TrainDateset(Dataset):

    def __init__(self, path, transform=None):
        
        # 参数
        class_name = ['F58001104805002120001',
                     'F58001104827202120004',
                     'F58001104821701010008',
                     'F58001104949503200002',
                     'F58000104805003240003',
                     'F58001104805003210004']
        
        para_angle_cls = 30 

        # 收集每类的文件名
        filename = []
        for class_i in range(len(class_name)):
            filename.append( os.listdir(osp.join(path,class_name[class_i])) )
        
        
        dataset_info = []
        shape_dicts = []
        for class_i in range(len(class_name)):
            # 对每一类收集信息
            shape_list = []
            angle_list = []
            rgb_list = []
            for i in range(len(filename[class_i])):
                # 解析文件名
                cls,xyz,rgb = filename[class_i][i].split("_")
                
                x,y,z = xyz.split(",")
                x,y,z = int(x),int(y),int(z)
                
                t = re.split("'|, |\[|\]",rgb)
                try:
                    r,g,b = float(t[3]),float(t[8]),float(t[13])
                except:
                    r,g,b = float(t[2]),float(t[5]),float(t[8])

                shape_list.append((x,y))
                angle_list.append(z)
                rgb_list.append((r,g,b))

            # 每个工件的xy取值集合都不同，对每个类别生成特定的形状标签
            shape_set = sorted(list(set(shape_list)))   # sorted 保证每次顺序一致  
            shape_dict = {shape_set[i]:i for i in range(len(shape_set))}
            shape_dicts.append(shape_dict)
            label_shape = np.array([shape_dict[item] for item in shape_list],dtype="int16")

            # 生成角度标签
            label_angle = (np.array(angle_list,dtype="int16")-1)//para_angle_cls  # angle_list 1~360

            # 整合信息
            for i in range(len(filename[class_i])):
                info_dict = {}
                info_dict["filename"] = filename[class_i][i]
                info_dict["xy"] = shape_list[i]
                info_dict["z"] = angle_list[i]
                info_dict["label_shape"] = label_shape[i]
                info_dict["label_angle"] = label_angle[i]
                info_dict["label_class"] = class_i
                info_dict["label_rgb"] = rgb_list[i]
                dataset_info.append(info_dict)

        # 生成全局形状类别并加入标签
        globalshape = [(item["label_class"],item["label_shape"]) for item in dataset_info]
        globalshape_set = sorted(list(set(globalshape)))
        globalshape_dict = {globalshape_set[i]:i for i in range(len(globalshape_set))}
        for item in dataset_info:
            item["label_gshape"] = globalshape_dict[(item["label_class"],item["label_shape"])]  
              
            
        self.dataset_info = dataset_info
        self.shape_dicts = shape_dicts
        self.globalshape_dict = globalshape_dict
        self.gshape2class = {v:k[0]  for (k,v) in globalshape_dict.items()}
        self.para_angle_cls = para_angle_cls
        self.class_name = class_name
        self.path = path
        self.filename = filename
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, index):
        info_dict = self.dataset_info[index]
        
        img_path = osp.join(self.path,self.class_name[info_dict["label_class"]],info_dict["filename"])
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        
        label = np.array([info_dict["label_class"],info_dict["label_shape"],
                          info_dict["label_angle"],info_dict["label_gshape"],
                         info_dict["label_rgb"][0],info_dict["label_rgb"][1],info_dict["label_rgb"][2]
                         ],
                         dtype="float32")
        
        if self.transform:
            img = self.transform(img)
        
        return img,label
    
class TestDateset(Dataset):

    def __init__(self, path, transform=None):
        
        # 参数
        class_name = ['F58001104805002120001',
                     'F58001104827202120004',
                     'F58001104821701010008',
                     'F58001104949503200002',
                     'F58000104805003240003',
                     'F58001104805003210004']

        # 收集每类的文件名
        filename = []
        for class_i in range(len(class_name)):
            filename.append( os.listdir(osp.join(path,class_name[class_i])) )
        
        
        dataset_info = []
        for class_i in range(len(class_name)):
            for i in range(len(filename[class_i])):
                info_dict = {}
                info_dict["filename"] = filename[class_i][i]
                info_dict["label_class"] = class_i
                dataset_info.append(info_dict)

        self.dataset_info = dataset_info
        self.class_name = class_name
        self.path = path
        self.filename = filename
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, index):
        info_dict = self.dataset_info[index]
        
        img_path = osp.join(self.path,self.class_name[info_dict["label_class"]],info_dict["filename"])
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        
        label = np.array([info_dict["label_class"]],dtype="float32")
        
        if self.transform:
            img = self.transform(img)
        
        return img,label
