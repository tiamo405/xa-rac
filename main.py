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
from model import LSTM
from src import write_txt, convert_input

class Model():
    def __init__(self, opt = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTM(input_size= 50, hidden_size= 128, num_layers= 4, num_classes= 2, device= self.device)
        self.checkpoint_model = os.path.join(opt.checkpoint_path, opt.name_model, opt.num_train, opt.num_ckp+'.pth')
        self.model.load_state_dict(torch.load(self.checkpoint_model)['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.replicate = opt.replicate


    def preprocess(self, points):
        if len(points) < self.replicate :
            for i in range(self.replicate - len(points)) :
                points.append(points[-1])
            points = np.array(points)
        else : 
            points = np.array(points)[-30:]
        points = np.reshape(points, (points.shape[0], points.shape[1]))
        return torch.tensor(points).to(self.device).unsqueeze(0).float()
    def predict(self, points):
        if len(points) ==0 :
            return 0
        points = self.preprocess(points)
        input = torch.tensor(points)
        label = self.model(input)
        label = np.argmax(label.cpu().detach().numpy())
        return label
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, choices=['camrtsp', 'webcam', 'video'])
    parser.add_argument('--camrtsp', type= str, default='none')
    parser.add_argument('--path_video', type= str, default='none')
    
    parser.add_argument('--tile', type=float, default= 1.2)
    parser.add_argument('--dTrash', type= float, default= 10)
    parser.add_argument('--checkpoint_path', type= str, default= 'checkpoints/')
    parser.add_argument('--replicate', type= int, default= 30)
    parser.add_argument('--name_model', type = str, default = 'LSTM')
    parser.add_argument('--num_train', type= str, default= '0')
    parser.add_argument('--num_ckp', type= str, default= 'best_epoch')
    opt = parser.parse_args()
    return opt
def main(opt, mot_tracker=None , yolo_person=None , \
         yolo_trash=None , model_pose= None ) :
    model = Model(opt= opt)
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
                pose = []
                try :
                    pose = model_pose(image)
                    if len(pose) != 0 :
                        points = np.array(pose[0])[:, 0:2]
                        
                        points = convert_input(points, frame_width, frame_height)
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
                print(model.preprocess(pose_id[id]).shape)
                label = model.predict(pose_id[id])
                print(label)
                label_id[id] = 'Xa rac' if label == 1 else 'binh thuong'
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f'{id}_{label_id[id]}', (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
            
        # cv2.imshow("Image", frame)
        cv2.imwrite('frame.jpg', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    print("done")

if __name__ == "__main__" : 
    opt = parse_opt()

    mot_tracker = Sort(max_age=10)
    yolo_person = torch.hub.load('yolov5', 'custom', path='checkpoints/yolov5n.pt', source='local')  # or yolov5n - yolov5x6, custom
    yolo_trash = torch.hub.load('yolov5', 'custom', path='checkpoints/trash.pt', source='local')  # or yolov5n - yolov5x6, custom
    model_pose = torch_openpose.torch_openpose('body_25')

    main(opt, mot_tracker = mot_tracker, yolo_person = yolo_person, \
         yolo_trash = yolo_trash, model_pose = model_pose)