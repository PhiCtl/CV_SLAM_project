# !/usr/bin/env python

import __init__
from cv_lib.object_detection import Object_Detection
from cv_lib.camera_listener import CameraListener
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import rospy


# topics list
topic_rs_status = '/RS_status'
topic_cv_status = '/CV_status'
topic_cv_datas = '/CV_datas'

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
        camera.get_frames()
        camera.get_info()

        # detect object
        detector.set_picture(camera.bgr_image)
        detector.get_mask(it = 2) # OK
        detector.find_centroids(threshold=1000)
        detector.get_pos(camera)
        detector.get_plane_orientation(camera, plot = False)

        for centroid_coo, plane_vector_coo in zip(detector.coo, detector.planes):
            publish(centroid_coo, plane_vector_coo)
        detector.reset()
        rate.sleep()


if __name__ == '__main__':
    run()