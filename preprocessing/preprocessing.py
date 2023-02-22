import cv2
import os
import torch
import sys
import numpy as np
import pandas as pd
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("src"))
sys.path.insert(0, root)
from cropvideo import crop_image_video
from src_pose import torch_openpose
from sort.sort import Sort 
class Preprocessing :
    def __init__(self, dir_data = 'data', parse = 'train',\
                 model_detect_person = None, model_open_pose = None, tracking = None):
        self.dir_data = dir_data
        self.parse = parse
        self.model_detect_person=model_detect_person
        self.model_open_pose= model_open_pose
        self.tracking = tracking
    def tien_xu_li(self):
        if not os.path.exists(os.path.join(self.dir_data, self.parse, 'images')) :
            os.mkdir(os.path.join(self.dir_data, self.parse, 'images'))
        name_videos = os.listdir(os.path.join(self.dir_data, self.parse, 'videos'))
        for name_video in name_videos :
            if name_video.split('.')[-1] in os.listdir(os.path.join(self.dir_data, self.parse, 'images')) :
                continue
            print('path video: ',os.path.join(self.dir_data, self.parse, 'videos', name_video))

            crop_image_video(path_video         = os.path.join(self.dir_data, self.parse, 'videos', name_video),\
                            path_save           = os.path.join(self.dir_data, self.parse, 'images'), \
                            model_detect_person = self.model_detect_person,\
                            model_open_pose     = self.model_open_pose,\
                            tracking            = self.tracking)
    def create_csv(self) :
        name_video = []
        ids = []
        labels = []
        for fname in os.listdir(os.path.join(self.dir_data, self.parse, 'images')):
            for id in os.listdir(os.path.join(self.dir_data, self.parse, 'images', fname)) :
                name_video.append(fname)
                ids.append(id)
                labels.append(0)
        df = pd.DataFrame({
            'name_video': name_video,
            'id' : ids,
            'label': labels
        })
        df.to_csv(os.path.join(self.dir_data, self.parse, 'labels.csv'), index=False)
if __name__ == "__main__":
    model_detect_person = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
    model_open_pose = torch_openpose.torch_openpose('body_25')
    tracking = Sort(max_age=10)
    prepro = Preprocessing(model_detect_person=model_detect_person, model_open_pose= model_open_pose, tracking= tracking)
    # prepro.tien_xu_li()
    prepro.create_csv()
    
        