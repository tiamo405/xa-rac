import numpy as np
import math
import cv2
import torch
import os

def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    
# draw the body keypoint and lims
def draw_bodypose(img, poses,model_type = 'body_25'):
    stickwidth = 4
    
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                   [0, 15], [15, 17]]
    njoint = 18
    if model_type == 'body_25':    
        njoint = 25
        limbSeq = [[1,8],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[10,11],\
                            [8,12],[12,13],[13,14],[1,0],[0,15],[15,17],[0,16],\
                                [16,18],[14,19],[19,20],[14,21],[11,22],[22,23],[11,24]]
    colors = [[0, 0, 153], [0, 51, 153], [0, 153, 102], [0, 102, 153], [0, 153, 153], \
            [0, 153, 51], [0, 153, 0], [51, 153, 0], [255,0, 0], [0, 255, 85],\
             [153, 102, 0], [153, 51, 0], [0, 170, 255], [51, 0, 153], [102, 0, 153],\
             [153, 0, 153], [153, 0, 102], [153, 0, 51], [85, 0, 255], [0, 0, 255], \
                [0, 0, 255],[0, 0, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255]]
    for i in range(njoint):
        for n in range(len(poses)):
            pose = poses[n][i]
            if pose[2] <= 0:
                continue
            x, y = pose[:2]
            cv2.circle(img, (int(x), int(y)), 6, colors[i], thickness=-1)
            # cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 

    for pose in poses:
        for limb,color in zip(limbSeq,colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <=0 or p2[2] <= 0:
                continue
            cur_canvas = img.copy()
            X = [p1[1],p2[1]]
            Y = [p1[0],p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.1, cur_canvas, 0.9, 0)
   
    return img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
