import cv2
import face_recognition
import numpy as np
import sys
from preprocessing.cropvideo import crop_image_video
from preprocessing.utils import point_object
import torch
from src import torch_openpose
model = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
tp = torch_openpose.torch_openpose('body_25')
crop_image_video(path_video="data/person.mp4", path_save="data/test1", \
    model_detect_person=model, model_open_pose= tp)
