#! /usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose
from std_msgs.msg import String

rospy.init_node('test_subscriber', anonymous=False)
pub = rospy.Publisher('/depth', String , queue_size=1)

def callback1(msg1): # bounding box
    try: ## no error = person detected
        print("Bbox x: ", msg1.detections[int(PPP)].bbox.center.x)
        print("Bbox y: ", msg1.detections[int(PPP)].bbox.center.y)
        box=msg1.detections[int(PPP)].bbox.center
        perspective(box)
    except: #error = no person. 
        print('No Person Detected')
    
def callback2(msg2): # confidence
    try:
        if len(msg2.data)!=0:
            print("confs: ", msg2.data[int(PPP)])
    except:
        print('No Person Detected')

def callback3(msg3): # class id
    global PPP # for only person
    PPP = 'fuck' # for only person
    if len(msg3.data)!=0:  # empty?
        for i in range(len(msg3.data)): # number of bbox
            if msg3.data[i]==0: # person class is number 0
                PPP = i
                print("clss: ", msg3.data[i])
    #os.system('cls' if os.name =='nt' else 'clear')

def perspective(box):
    ## setting
    print("join")
    returnw = 416
    returnh = 416
    c2c = 100    #cone to cone real distance
    
    Left_Top = [120,0]
    Right_Top = [520,0]
    Left_Bot = [20, 480]
    Right_Bot = [620, 480]
    bbox = (box.x,box.y,1)
    print(bbox)
    ar1 = np.float32([[Left_Top], [Right_Top], [Left_Bot], [Right_Bot]])
    ar2 = np.float32([[0,0],[returnw,0],[0,returnh],[returnw,returnh]])
    mtx = cv2.getPerspectiveTransform(ar1,ar2)
    print(mtx)
    data = np.dot(bbox,mtx)
    pix_depth = returnh - data[1]/data[2]
    depth= pix_depth*c2c/abs(Right_Bot[0]-Left_Bot[0])
    pub.publish(depth)
    print("test", depth)


def listener():
    rospy.Subscriber('/clss', Float32MultiArray, callback3) # class id , detecting only person
    rospy.Subscriber('/confs', Float32MultiArray, callback2) # confidence
    rospy.Subscriber('/bbox', Detection2DArray, callback1) # bouding box center x,y

        
if __name__ == '__main__':
    listener()
    rospy.spin()
