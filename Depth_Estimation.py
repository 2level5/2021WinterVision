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
        global box
        print("Bbox x: ", msg1.detections[int(PPP)].bbox.center.x)
        print("Bbox y: ", msg1.detections[int(PPP)].bbox.center.y)
        box=msg1.detections[int(PPP)].bbox.center
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

def perspective():
    ## setting
    returnw = 416
    returnh = 416
    ##
    Left_Top = [100,1]
    Right_Top = [200,1]
    Left_Bot = [100, 200]
    Right_Bot = [200, 200]

    ar1 = np.float32([[Left_Top],[Right_Top],[Left_Bot],[Right_Bot]])
    ar2 = np.float32([0,0],[retrunw,0],[0,returnh],[returnw,returnh])
    mtx = cv2.getPerspectiveTransform(ar1,ar2)

    data = cv2.warpPerspective(box,mtx,(returnw,returnh))
    pub.publish(data)



def listener():
    rospy.Subscriber('/clss', Float32MultiArray, callback3) # class id , detecting only person
    rospy.Subscriber('/confs', Float32MultiArray, callback2) # confidence
    rospy.Subscriber('/bbox', Detection2DArray, callback1) # bouding box center x,y

        
if __name__ == '__main__':
    listener()
    rospy.spin()