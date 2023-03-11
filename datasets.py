from os import path as osp
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import os
import json 
import numpy as np
from preprocessing.util import read_json
class DatasetLSTM(data.Dataset):
    
    def __init__(self, opt):
        def myFunc(e):
            return int(e.split(".")[0])
        super(DatasetLSTM, self).__init__()
        self.replicate = opt.replicate
        self.data_path = os.path.join(opt.train_dir, "images")
        self.label_path = os.path.join(opt.train_dir, "labels.csv")
        self.myFunc = myFunc
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # load data list
        json_s = []
        image_s = []
        labels = []
        df  = pd.read_csv(self.label_path)
        for i in range(len(df)):
            input_json = []
            input_img = []
            vid_name = str(df['name_video'][i]).zfill(4)
            id = df['id'][i]
            label = df['label'][i]
            path_json = os.path.join(self.data_path, vid_name, 'json', str(id).zfill(4))
            path_img = os.path.join(self.data_path, vid_name, 'img', str(id).zfill(4))
            fname_json_s = os.listdir(path_json)
            for fname in fname_json_s :
                pose_data = read_json(os.path.join(path_json, fname))
                input_json.append(pose_data)
            if len(input_json) ==0 :
                continue
            if len(input_json) < self.replicate :
                for i in range(self.replicate - len(input_json)) :
                    input_json.append(input_json[-1])
                input_json = np.array(input_json)
            else : 
                input_json = np.array(input_json)[-self.replicate:]
            # input_json = np.array(input_json).reshape(self.replicate * 50)

            fname_img_s = os.listdir(path_img)
            for fname in fname_img_s :
                # pose_data = read_json(os.path.join(path_json, fname))
                input_img.append(Image.open(os.path.join(path_img, fname)))
            
            image_s.append(input_img)
            labels.append(label)
            json_s.append(input_json)


        self.json_s = json_s
        self.labels = labels
        self.image_s = image_s
 
    def __getitem__(self, index):
        label = self.labels[index]
        # input_json = np.array(self.json_s[index]).reshape(1, len(self.json_s[index]))
        input_json = np.array(self.json_s[index])
        # print(input_json.shape)
        input_json = np.reshape(input_json, (input_json.shape[0], input_json.shape[1]))
        input_img = self.image_s[index]
        result = {
            'label' : label,
            'input_json': input_json,
            # 'input_img' : input_img
        }
        
        return result

    def __len__(self):
        return len(self.labels)
    