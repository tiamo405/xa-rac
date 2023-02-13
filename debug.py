import cv2
import face_recognition
import numpy as np
import sys
# sys.path.insert(0, "sort")
# from sort.sort import Sort  # noqa
from preprocessing.cropvideo import crop_image_video
from preprocessing.utils import point_object
import torch
from src import torch_openpose
a = np.array([[0,1],[2,3],[4,5]])
tmp = list(np.array(a).reshape(6))
print(tmp)
model = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
tp = torch_openpose.torch_openpose('body_25')
# crop_image_video(path_video="data/person.mp4", path_save="data/test1", \
#     model_detect_person=model, model_open_pose= tp)
image = cv2.imread("data/2nguoi.jpg")
results = model(image)
person_points = point_object(results)
print(person_points)
for (l, t, r, b) in (person_points) :
    img = image[t:b, l:r]
    print(tp(img))
