from os import path as osp
from torch.utils import data
from torchvision import transforms
import cv2
import pandas as pd
import os
import json 
import numpy as np
class DatasetLSTM(data.Dataset):
    
    def __init__(self, opt):
        def myFunc(e):
            return int(e.split(".")[0])
        super(DatasetLSTM, self).__init__()
        self.replicate = opt.replicate
        self.data_path = osp.join(opt.train_dir, "json")
        self.label_path = osp.join(opt.train_dir, "label.csv")
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
            vid_name = str(df['fname'][i])
            label = df['labels'][i]
            f_json = os.listdir(os.path.join(self.data_path, vid_name))
            f_json.sort(key = self.myFunc)
            for f_name in f_json:
                with open(osp.join(self.data_path, vid_name, f_name), 'r') as f:
                    pose_label = json.load(f)
                    pose_data = pose_label['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1, 3))[:, :2].reshape(50)
                    
                input.append(pose_data)
            for j in range(len(input), self.replicate):
                input.append(input[-1])
            labels.append(label)
            input = np.array(input).reshape(1500)
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
    