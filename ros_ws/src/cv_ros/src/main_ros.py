# !/usr/bin/env python

import __init__
from cv_lib.object_detection import Object_Detection
from cv_lib.camera_listener import CameraListener
from geometry_msgs.msg import PoseStamped
import cv2
import rospy
import numpy as np

def init_node():
    global msg, pub, rate
    rospy.init_node("Vision")
    pub = rospy.Publisher('/Vision', PoseStamped, queue_size=10)
    rate = rospy.Rate(10)
    msg = PoseStamped()

def publish(centroid_coo, plane_vector_coo):
    msg.header.frame_id, msg.header.stamp = "camera", rospy.Time.now()
    # [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] = centroid_coo
    msg.pose.position.x = centroid_coo[0]/1000
    msg.pose.position.y = centroid_coo[1]/1000
    msg.pose.position.z = centroid_coo[2]/1000

    [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] = plane_vector_coo
    # rospy.loginfo(msg)
    pub.publish(msg)
    # rate.sleep()

def run():
    init_node()

    # start camera listener and detector
    camera = CameraListener()
    detector = Object_Detection()

    while not rospy.is_shutdown():
        rospy.loginfo("Begin main loop")
        camera.get_frames()
        camera.get_info()

        # detect object
        detector.set_picture(camera.bgr_image)
        detector.get_mask(it = 2) # OK
        detector.find_centroids(threshold=1000)
        detector.get_pos(camera)
        detector.get_plane_orientation(camera, plot = False)

        rospy.loginfo("Finish computation, will publish")
        nb_msgs=0
        for centroid_coo, plane_vector_coo in zip(detector.coo, detector.planes):
            publish(centroid_coo, plane_vector_coo)
            nb_msgs += 1
        rospy.loginfo("Published {} messages, will sleep".format(nb_msgs))
        rate.sleep()

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
    run()