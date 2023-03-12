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
import random
pwd = os.path.dirname(os.path.realpath("sort"))
sys.path.insert(0, root)
from preprocessing.util import point_object
from sort.sort import Sort
from preprocessing.util import draw
from model import LSTM
from src import write_txt, convert_input, str2bool

class Model():
    def __init__(self, opt = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTM(input_size= 50, hidden_size= 128, num_layers= 4, num_classes= 2, device= self.device)
        self.checkpoint_model = os.path.join(opt.checkpoint_path, opt.name_model, opt.num_train, opt.num_ckpt+'.pth')
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
    parser.add_argument('--option', type=str, choices=['camrtsp', 'webcam', 'video'], default= 'video')
    parser.add_argument('--camrtsp', type= str, default='none')
    parser.add_argument('--path_video', type= str, default='data/test/split_video000.mp4')

    parser.add_argument('--tile', type=float, default= 1.2)
    parser.add_argument('--dTrash', type= float, default= 10)
    parser.add_argument('--checkpoint_path', type= str, default= 'checkpoints/')
    parser.add_argument('--replicate', type= int, default= 300)
    parser.add_argument('--name_model', type = str, default = 'LSTM')
    parser.add_argument('--num_train', type= str, default= '0')
    parser.add_argument('--num_ckpt', type= str, default= 'best_epoch')
    parser.add_argument('--save_video', type= str2bool, default= True)
    parser.add_argument('--path_save_video', type= str, default='results/video')
    opt = parser.parse_args()
    return opt

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


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
    fps = cap.get(cv2.CAP_PROP_FPS)
    path_save_video = os.path.join(opt.path_save_video, str(len(os.listdir(opt.path_save_video))).zfill(4)+ '.avi')
    video = cv2.VideoWriter(path_save_video, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, (frame_width, frame_height))
    # vid_path, vid_writer = [None] * 1, [None] * 1
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
                print((left, top, right, bottom))
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
                # print(model.preprocess(pose_id[id]).shape)

                if (id == 1):
                    opt.replicate = 100
                elif (id == 2):
                    opt.replicate = 110
                else:
                    opt.replicate = 60
                # print(f"id: {id} --- {len(pose_id[id])}")
                # if len(pose_id[id]) >= opt.replicate :
                #     # label = model.predict(pose_id[id])
                #     label = 1
                #     print(label)
                #     label_id[id] = 'Xa rac' if label == 1 else 'binh thuong'
                #     # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                #     # font = cv2.FONT_HERSHEY_DUPLEX
                #     # cv2.putText(frame, f'{id}_{label_id[id]}', (left + 6, top + 6), font, 1, (0, 0, 255), 1)
                #     # plot_one_box([left, top, right, bottom], frame, (0, 0, 255), f'{id}_{label_id[id]}', 3)
                # else:
                #     label_id[id] = 'binh thuong'
                    # plot_one_box([left, top, right, bottom], frame, (0, 237, 255), f'{id}_{label_id[id]}', 3)
                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 237, 255), 3)
                    # font = cv2.FONT_HERSHEY_DUPLEX
                    # cv2.putText(frame, f'{id}_{label_id[id]}', (left + 6, top + 6), font, 1, (0, 0, 255), 1)

        # cv2.imshow("Image", frame)
        if opt.save_video == True :
            video.write(frame)
            # fps, w, h = 30, frame.shape[1], frame.shape[0]
            # save_path = path_save_video
            # i = 0
            # if vid_path[i] != save_path:  # new video
            #     vid_path[i] = save_path
            #     if isinstance(vid_writer[i], cv2.VideoWriter):
            #         vid_writer[i].release()  # release previous video writer
            #     # if vid_cap:  # video
            #     vid_cap = cap
            #     fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #     vid_writer[i] = cv2.VideoWriter(save_path.split('.')[0], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            # vid_writer[i].write(frame)

        cv2.imwrite('frame.jpg', frame)
        # break
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