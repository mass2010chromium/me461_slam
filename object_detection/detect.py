import os
import sys
import cv2
import time
import numpy as np
import json
sys.path.append(os.path.expanduser('~/me461_slam'))
from utils.read_mat import SharedArray

videocap = SharedArray("../.webserver.video", [480, 640, 3], dtype=np.uint8)

whT = 160 #320
confThreshold =0.5
nmsThreshold= 0.2
play = True

previous = ''


classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Undistort 
with open("intrinsics.json") as _intrinsics:
    intrinsics = json.load(_intrinsics)
camera_mat = np.array(intrinsics["matrix"])
camera_dist = np.array(intrinsics["distortion"])


## Model Files
modelConfiguration = "yolov3-tiny.cfg" #yolov3-tiny.cfg
modelWeights = "yolov3-tiny.weights" #yolov3-tiny.weights
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int(det[0]*wT-w/2),int(det[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classID)
                confs.append(float(confidence))
    
    indices = list(cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold))

    return indices, bbox, confs, classNames, classIds
'''
    for i in indices:
        i = int(i)
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
'''
    
while True:
    time0 = time.time()

    success, image = videocap.read()

    blob = cv2.dnn.blobFromImage(image, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)


    detect = findObject(outputs,image)



    for i in detect[0]:
        i = int(i)
        box = detect[1][i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(image,f'{detect[3][detect[4][i]].upper()} {int(detect[2][i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if detect[3][detect[4][i]] != previous:
            cmd = 'espeak "{}"'.format(detect[3][detect[4][i]])
            os.system(cmd)
        previous = detect[3][detect[4][i]]

        
        
    #Undistort
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #out = cv2.undistort(gray, camera_mat, camera_dist)


    cv2.imshow("video", image) #image
    cv2.waitKey(1)
    time1 = time.time()
    print(1/(time1-time0)) #fps
