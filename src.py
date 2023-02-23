import cv2
import torch
import os
import numpy as np

def write_txt(noidung, path, remove = False):
    if os.path.exists(path) and remove == True:
        os.remove(path)
    with open(path, 'a') as f:
        f.write("%s\n" % noidung)
        f.close()

def convert_input(points, width, height) :
    tmp = list(points.reshape(50))
    arr = []
    for i in range(len(tmp)) :
        if i % 2 == 0 :
            arr.append(tmp[i] / width)
        elif i%2 ==1:
            arr.append(tmp[i]/ height)
    return np.array(arr)