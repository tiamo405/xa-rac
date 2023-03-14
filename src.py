import cv2
import torch
import os
import numpy as np
import argparse
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

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')