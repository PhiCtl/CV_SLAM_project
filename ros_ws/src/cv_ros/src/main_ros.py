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
    global msg, cv_status, pub, pub_state, rate
    rospy.loginfo("init_node")
    rospy.init_node("Vision")
    pub = rospy.Publisher(topic_cv_datas, PoseStamped, queue_size=10)
    pub_state = rospy.Publisher(topic_cv_status, String, queue_size = 10)
    rate = rospy.Rate(10)
    msg = PoseStamped()
    cv_status = String()

def publish(centroid_coo, plane_vector_coo):
    msg.header.frame_id, msg.header.stamp = "camera", rospy.Time.now()
    # Convert from mm to m
    msg.pose.position.x = centroid_coo[0]/1000
    msg.pose.position.y = centroid_coo[1]/1000
    msg.pose.position.z = centroid_coo[2]/1000

    [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] = plane_vector_coo

    pub.publish(msg)


def run():
    init_node()

    # start camera listener and detector
    camera = CameraListener()
    detector = Object_Detection()
    cv_status.data = 'IDLE'
    while not rospy.is_shutdown():

        # Wait for Robotic station status
        rs_status = rospy.wait_for_message(topic_rs_status, String) # will be always active, otherwise code stops here
        if rs_status.data == 'STANDSTILL':

            # Take picture
            camera.get_frames()
            camera.get_info()
            # detect object
            detector.set_picture(camera.bgr_image)
            detector.get_mask(it = 2) # OK
            if detector.find_centroids(threshold=10000):
                # Find positions and orientations
                detector.get_pos(camera)
                detector.get_plane_orientation(camera, plot = False)

                # status
                cv_status.data = 'SUCCESS'

                # publish
                for centroid_coo, plane_vector_coo in zip(detector.coo, detector.planes):
                    publish(centroid_coo, plane_vector_coo)
                detector.reset()
            else:
                cv_status.data = 'RETRY'
        pub_state.publish(cv_status)
        rate.sleep()

if __name__ == '__main__':
    run()