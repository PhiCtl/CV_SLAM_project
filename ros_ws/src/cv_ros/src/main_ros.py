#!/usr/bin/env python

import __init__
from cv_lib.object_detection import Object_Detection
from camera_listener import Camera_Listener
from cv_ros.msg import ObjectPos
import cv2
import numpy as np
import rospy

def init_node():
    global pub, rate, msg

    rospy.init_node("Vision")
    pub = rospy.Publisher('/Vision', ObjectPos)
    rate = rospy.Rate(10)
    msg = ObjectPos()

def publish():
    msg.centroid = np.array([0,0,0])
    msg.plane_vector = np.array([1,1,1])
    rospy.loginfo(msg)
    pub.publish(msg)
    rate.sleep()

def run():
    pass

def test():
    camera = Camera_Listener()
    camera.get_frames()
    img = camera.bgr_image
    cv2.imshow('img test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_node()
    test()