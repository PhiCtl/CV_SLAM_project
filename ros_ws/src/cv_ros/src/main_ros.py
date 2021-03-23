# !/usr/bin/env python

import __init__
from cv_lib.object_detection import Object_Detection
from cv_lib.camera_listener import CameraListener
from cv_ros.msg import ObjectPos
import cv2
import rospy
import numpy as np

def init_node():
    global msg, pub, rate
    rospy.init_node("Vision")
    pub = rospy.Publisher('/Vision', ObjectPos, queue_size=10)
    rate = rospy.Rate(10)
    msg = ObjectPos()

def publish(centroid_coo, plane_vector_coo):
    msg.centroid = centroid_coo
    msg.plane_vector = plane_vector_coo
    rospy.loginfo(msg)
    pub.publish(msg)
    rate.sleep()

def run():
    init_node()

    # start camera listener
    camera = CameraListener()
    camera.get_frames()

    # detect object
    detector = Object_Detection(camera.bgr_image)
    detector.get_mask(plot = True)
    #detector.find_centroids(threshold=100)
    #detector.get_pos(camera)
    #detector.get_plane_orientation(camera, plot=True)

    #for centroid_coo, plane_vector_coo in zip(detector.coo, detector.planes):
     #   publish(centroid_coo, plane_vector_coo)


def test_listener():
    camera = CameraListener()
    camera.get_frames()
    img = camera.bgr_image
    cv2.imshow('img test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_talker():
    publish()

if __name__ == '__main__':
    #run()
    init_node()
    test_listener()