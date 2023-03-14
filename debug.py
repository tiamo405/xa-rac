import cv2
import face_recognition
import numpy as np
import sys
# from preprocessing.cropvideo import crop_image_video
from preprocessing.util import point_object, draw
import torch
from src_pose import torch_openpose
import time
import datetime
import os
def check_model_trash() :
    model = torch.hub.load('yolov5', 'custom', path='checkpoints/trash.pt', source='local')
    im = cv2.imread("data/lonbia.jpg")
    arr = point_object(model(im), label=[0,1,2,3,4,5,6])
    for a in arr :
        print(a[1])
def check_dict():
    label = {}
    map1 =[100*[]]
    for i in range(5) :
        arr = []
        for j in range(10) :
            arr.append(j)
        if 1 not in label:
            label[1] = [arr]
        else :
            label[1].append(arr)
    print(label)
def numpyarray() :
    a = []
    for i in range(50) :
        a.append(i)
    a = np.array(a)
    a = np.append(a, 100)
    print(a[-30:])
def convertint2str():
    a = 2
    print(str(a).zfill(4))
    print(str(1).zfill(2))

def checkmodelpose() :
    model_open_pose = torch_openpose.torch_openpose('body_25')
    image = cv2.imread("data/train/images/0000/img/0001/0001.jpg")
    start_time = time.time()
    pose = model_open_pose(image)
    print(pose)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(total_time)

def checksort() :
    fnames = os.listdir("data/train/images")
    fnames = sorted(fnames)
    print(fnames)

if __name__ =="__main__":
    # check_model_trash()
    # check_dict()
    # numpyarray()
    # convertint2str()
    checkmodelpose()

