import cv2
import os
import json
import numpy as np

import sys
from pathlib import Path

root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("sort"))
sys.path.insert(0, root)
from sort.sort import Sort 

from preprocessing.util import point_object, search_id, draw, createjson
from src import write_txt

def crop_image_video(path_video, path_save, label, model_detect_person, model_open_pose) :
    
    name_video = path_video.split('/')[-1].split('.')[0]
    print(f'name video : {name_video}')


    path_save_image = os.path.join(path_save, 'rbg-images', label, name_video)
    path_save_txt = os.path.join(path_save, 'txt', label, name_video)
    path_save_json = os.path.join(path_save, 'json', label, name_video)

    if not os.path.exists(path_save_image) :
        os.makedirs(path_save_image)
    if not os.path.exists(path_save_txt) :
        os.makedirs(path_save_txt)
    if not os.path.exists(path_save_json) :
        os.makedirs(path_save_json)

    cap = cv2.VideoCapture(path_video)
    mot_tracker = Sort(max_age= 10)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ids = None
    results = None
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
                id = int(id)
                image = frame[top:bottom, left: right]
                if bottom - top < 20 or right - left < 20 :
                    continue
                path_save_img_id = os.path.join(path_save_image, (str(id).zfill(4)))
                path_save_json_id = os.path.join(path_save_json, (str(id).zfill(4)))
                path_save_boxtxt_id = os.path.join(path_save_txt, (str(id).zfill(4)))
                if not os.path.exists(path_save_img_id) :
                    os.mkdir(path_save_img_id)
                if not os.path.exists(path_save_json_id) :
                    os.mkdir(path_save_json_id)
                
                id_frame = search_id(path_save_img_id)
                cv2.imwrite(os.path.join(path_save_img_id, str(id_frame).zfill(4) + '.jpg'), image)
                if label == 'trashDumping' :
                    write_txt(noidung= '1 {} {} {} {}'.format(left, top, right, bottom),path= path_save_boxtxt_id)
                else :
                    write_txt(noidung= '0 {} {} {} {}'.format(left, top, right, bottom),path= path_save_boxtxt_id)
                print(f'save image: name video: {name_video}, id:{id}, id_frame:{id_frame}')
                try:
                    pose = model_open_pose(image)
                    
                    # cv2.imwrite(os.path.join(path_save_img_json, str(id_frame) + '.jpg'), image)
                    if len(pose) > 0 : # detect pose
                        createjson(poses= pose, \
                            path_json= os.path.join(path_save_json_id, str(id_frame).zfill(4)+'.json'),
                            width= right - left, height= bottom - top)
                        print(f'save json: name video: {name_video}, id:{id}, id_frame:{id_frame}')
                        image = draw(image, pose)    
                        cv2.imwrite(os.path.join(path_save_img_id, str(id_frame).zfill(4) + '.jpg'), image)
                except :
                    continue
        # cv2.imshow("Image", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    print("done")
