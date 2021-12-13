import cv2
import numpy as np
import urllib.request
import json
import math

import sys

from utils.read_mat import SharedArray
from motion.image_cspace import ImageCSpace

with open("map_info.json", "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2
cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)

while True:
    
