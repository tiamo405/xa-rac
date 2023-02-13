import cv2
import os
import json
import numpy as np
from src import torch_openpose
import sys
sys.path.insert(1, "sort")
sys.path.insert(0, "preprocessing/utils")
from sort.sort import Sort 

from .utils import point_object, search_id
pwd = os.path.dirname(os.path.realpath(__file__))

def createjson(poses, path_json, width, height):
    tmp = list(np.array(poses[0]).reshape(75))
    arr = []
    for i in range(len(tmp)) :
        if i % 3 == 0 :
            arr.append(tmp[i] / width)
        elif i%3 ==1:
            arr.append(tmp[i]/height)
    data_dict={
        # "pose_keypoints_2d":list(np.array(poses[0]).reshape(75)),
        "pose_keypoints_2d":arr,
        'width' : width,
        'height': height
    }
    data_string = json.dumps(data_dict)
    myjsonfile = open(path_json, "w")
    myjsonfile.write(data_string)
    myjsonfile.close()

def crop_image_video(path_video, path_save, model_detect_person, model_open_pose) :
    name_video = path_video.split('/')[-1]
    if not os.path.exists(path_save) :
        os.mkdir(path_save)
    cap = cv2.VideoCapture(path_video)
    mot_tracker = Sort(max_age=10)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while(cap.isOpened()) :
        ret, frame = cap.read()
        if frame is None:
            break
        results = model_detect_person(frame)
        person_locations = point_object(results)
        if len(person_locations) != 0:
            dets_list = [[l, t, r, b, 1] for (l, t, r, b) in person_locations]
            dets = np.array(dets_list)
            trackers = mot_tracker.update(dets)
            ids = trackers[:, 4].flatten()
            for (left, top, right, bottom), id in zip(person_locations, ids):
                image = frame[top:bottom, left: right]
                path_save_img_json = os.path.join(path_save, str(id))
                if not os.path.exists(path_save_img_json) :
                    os.mkdir(path_save_img_json)
                id_frame = search_id(path_save_img_json)
                cv2.imwrite(os.path.join(path_save_img_json, str(id_frame) + '.jpg'), image)
                try:
                    pose = model_open_pose(image)
                    print(pose)
                    if len(pose) > 0 :
                        createjson(poses= pose, \
                            path_json= os.path.join(path_save_img_json, str(id_frame)+'.json'),
                            width= right - left, height= bottom - top)
                except :
                    continue
        # cv2.imshow("Image", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    print("done")
