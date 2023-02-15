import cv2
import face_recognition
import numpy as np
import sys
from preprocessing.cropvideo import crop_image_video
from preprocessing.util import point_object, draw
import torch
from src import torch_openpose
from sort.sort import Sort
model = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
tp = torch_openpose.torch_openpose('body_25')
tracking  = Sort(max_age= 10)
# crop_image_video(path_video="data/person.mp4", path_save="data/test1", \
#     model_detect_person=model, model_open_pose= tp, tracking= tracking)
im = cv2.imread("data/test1/person/1.0/1.jpg")
pose = tp(im)
print(pose)
image = draw(im, pose)
cv2.imwrite('debug.jpg', image)