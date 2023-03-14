import cv2
import os
import torch
import sys
import numpy as np
import pandas as pd
import argparse
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("src"))
sys.path.insert(0, root)
from cropvideo import crop_image_video
from src_pose import torch_openpose
from sort.sort import Sort 
class Preprocessing :
    def __init__(self, opt):
        self.model_detect_person = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
        self.model_open_pose = torch_openpose.torch_openpose('body_25')
        self.dir_data = opt.dir_data
        self.phase = opt.phase
        self.label = opt.label
        self.folder_videos = opt.folder_videos
    def crop_video(self):
        
        folder_save = os.path.join(self.dir_data, self.phase, 'dataset')
        if not os.path.exists(folder_save) :
            os.makedirs(folder_save)
        if not os.path.exists(os.path.join(folder_save, 'rbg-images', self.label)) :
            os.makedirs(os.path.join(folder_save, 'rbg-images', self.label))
        if not os.path.exists(os.path.join(folder_save, 'txt', self.label)) :
            os.makedirs(os.path.join(folder_save, 'txt', self.label))
        if not os.path.exists(os.path.join(folder_save, 'json', self.label)) :
            os.makedirs(os.path.join(folder_save, 'json', self.label))
    
        name_videos = os.listdir(self.folder_videos)
        name_videos = sorted(name_videos)

        for name_video in name_videos :
            if name_video.split('.')[-1] in os.listdir(os.path.join(self.dir_data, self.phase, 'dataset')) :
                continue
            print('path video: ',os.path.join(self.folder_videos, name_video))
            
            crop_image_video(path_video         = os.path.join(self.folder_videos, name_video),\
                            path_save           = folder_save, \
                            label               = self.label,
                            model_detect_person = self.model_detect_person,\
                            model_open_pose     = self.model_open_pose)
            
    def create_csv(self) :
        name_videos = []
        ids = []
        labels = []
        fnames = os.listdir(os.path.join(self.dir_data, self.phase, 'dataset'))
        fnames = sorted(fnames)
        for name_video in fnames:
            for id in sorted(os.listdir(os.path.join(self.dir_data, self.phase, 'dataset', name_video, 'img'))) :
                name_videos.append(name_video)
                ids.append(id)
                labels.append(1)
        df = pd.DataFrame({
            'name_video': name_videos,
            'id' : ids,
            'label': labels
        })
        df.to_csv(os.path.join(self.dir_data, self.parse, 'labels.csv'), index=False)
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default= 'data')
    parser.add_argument('--phase', type= str, default='train')
    parser.add_argument('--folder_video', type= str, default='data/train/gdg')
    parser.add_argument('--label', type=str, default='trashDumping')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":

    opt = get_opt()
    print('\n'.join(map(str,(str(opt).split('(')[1].split(',')))))

    prepro = Preprocessing(opt)

    prepro.crop_video()
    # prepro.create_csv()
    
        