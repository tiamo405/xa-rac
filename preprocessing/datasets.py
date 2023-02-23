from os import path as osp
from torch.utils import data
from torchvision import transforms
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
        inputs = []
        labels = []
        df  = pd.read_csv(self.label_path)
        for i in range(len(df)):
            input = []
            vid_name = str(df['name_video'][i])
            id = df['id'][i]
            label = df['label'][i]
            path = os.path.join(self.data_path, vid_name, str(id)+'.0')
            fnames = os.listdir(path)
            for fname in fnames :
                if '.json' in fname :
                    pose_data = read_json(os.path.join(path, fname))
                    input.append(pose_data)
            if len(input) ==0 :
                continue
            if len(input) < self.replicate :
                for i in range(self.replicate - len(input)) :
                    input.append(input[-1])
                input = np.array(input)
            else : 
                input = np.array(input)[-30:]
            input = np.array(input).reshape(self.replicate * 50)
            labels.append(label)
            inputs.append(input)

        self.inputs = inputs
        self.labels = labels
 
    def __getitem__(self, index):
        label = self.labels[index]
        input = np.array(self.inputs[index]).reshape(1, len(self.inputs[index]))

        result = {
            'label' : label,
            'input': input,
        }
        
        return result

    def __len__(self):
        return len(self.labels)
    