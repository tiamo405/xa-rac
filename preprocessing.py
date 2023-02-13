import cv2
import os
import numpy as np

class preprocessing :
    def __init__(self, dir_data = 'data', parse = 'train'):
        self.dir_data = dir_data
        self.parse = parse

    
        