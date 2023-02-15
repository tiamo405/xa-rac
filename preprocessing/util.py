import numpy as np
import os
import cv2
def point_object(res) :
    person_locations = []
    for i in range(len(res.xyxy[0])):
        if int(res.xyxy[0][i][5]) == 0:
            l, t, r, b=int(res.xyxy[0][i][0]), int(res.xyxy[0][i][1]),\
            int(res.xyxy[0][i][2]), int(res.xyxy[0][i][3])
            if r - l > 50 and b - t > 50 :
                person_locations.append((l,t,r,b))
    return person_locations
def search_id(path) :
    fnames = os.listdir(path)
    dem = 0
    for fname in fnames :
        if '.jpg' in fname or '.png' in fname :
            dem += 1
    return dem + 1

def draw(image, pose) :
    BODY_25_LINES = [
    [17, 15, 0, 1, 8, 9, 10, 11, 22, 23],  # Right eye down to right leg
    [11, 24],  # Right heel
    [0, 16, 18],  # Left eye
    [4, 3, 2, 1, 5, 6, 7],  # Arms
    [8, 12, 13, 14, 19, 20],  # Left leg
    [14, 21]  # Left heel
]
    points = np.array(pose[0])[:, 0:2]
    # for (x,y) in (points) :
        
    # print(points)
    for lines in BODY_25_LINES :
        for i in range(len(lines)-1):
            (xpt1, ypt1) = points[lines[i]]
            (xpt2, ypt2) = points[lines[i+1]]
            if (xpt1, ypt1) == (0, 0) or (xpt2, ypt2) == (0, 0):
                continue
            cv2.line(image, (int(xpt1), int(ypt1)), (int(xpt2), int(ypt2)), color= (255,0,0), thickness= 1)
    return image

        
    