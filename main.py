import cv2
import pandas
from model import LSTM
from src_pose import torch_openpose,util
import numpy as np
import json
import os
import torch
import sys
import argparse
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("sort"))
sys.path.insert(0, root)
from preprocessing.util import point_object
from sort.sort import Sort
from preprocessing.util import draw

mot_tracker = Sort(max_age=10)
yolo_person = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
yolo_trash = torch.hub.load('yolov5', 'custom', path='checkpoints/trash.pt', source='local')  # or yolov5n - yolov5x6, custom
model_pose = torch_openpose.torch_openpose('body_25')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, choices=['camrtsp', 'webcam', 'video'])
    parser.add_argument('--camrtsp', type= str, default='none')
    parser.add_argument('--path_video', type= str, default='none')
    
    parser.add_argument('--tile', type=float, default= 1.2)
    parser.add_argument('--dTrash', type= float, default= 10)
    opt = parser.parse_args()
    return opt
def main(opt, mot_tracker = mot_tracker, yolo_person = yolo_person, \
         yolo_trash = yolo_trash, model_pose = model_pose) :
    if opt.option == 'camrtsp' :
        cap = cv2.VideoCapture(opt.camrtsp)
    elif opt.option == 'webcam' :
        cap = cv2.VideoCapture(0)
    else :
        cap = cv2.VideoCapture(opt.path_video)
    label_id = {}
    pose_id = {}
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        results = yolo_person(frame)
        person_locations = point_object(results)
        if len(person_locations) != 0:
            dets_list = [[l, t, r, b, 1] for (l, t, r, b) in person_locations]
            dets = np.array(dets_list)
            trackers = mot_tracker.update(dets)
            ids = trackers[:, 4].flatten()
            for (left, top, right, bottom), id in zip(person_locations, ids):
                label_id[id] = 'binh thuong'
                image = frame [top: bottom, left :right]
                try :
                    pose = model_pose(image)
                    if len(pose) != 0 :
                        points = np.array(pose[0])[:, 0:2]
                        frame = draw(frame, pose, left, top)
                        # print(points)
                        if id not in pose_id:
                            pose_id[id] = [points]
                        else:
                            pose_id[id].append(points)
                        # print(pose_id[id])
                         
                except :
                    print('error')
                    continue
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f'{id}_{label_id[id]}', (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
            
        # cv2.imshow("Image", frame)
        cv2.imwrite('frame.jpg', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    print(label_id)
    print(pose_id)
    print("done")

if __name__ == "__main__" : 
    opt = parse_opt()
    
    main(opt, mot_tracker = mot_tracker, yolo_person = yolo_person, \
         yolo_trash = yolo_trash, model_pose = model_pose)