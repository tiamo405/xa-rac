import numpy
import os
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
            dem+=1
    return dem+1
