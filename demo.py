from src import torch_openpose,util
import numpy as np
import cv2
import json
import os
import torch
import sys
sys.path.insert(0, "preprocessing")
sys.path.insert(0, "sort")
from preprocessing.util import point_object
from sort.sort import Sort

mot_tracker = Sort(max_age=10)
model = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
tp = torch_openpose.torch_openpose('body_25')

cap = cv2.VideoCapture("data/person.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    results = model(frame)
    person_locations = point_object(results)
    if len(person_locations) != 0:
        dets_list = [[l, t, r, b, 1] for (l, t, r, b) in person_locations]
        dets = np.array(dets_list)
        trackers = mot_tracker.update(dets)
        ids = trackers[:, 4].flatten()
        for (left, top, right, bottom), id in zip(person_locations, ids):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(id), (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
    # cv2.imshow("Image", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
print("done")