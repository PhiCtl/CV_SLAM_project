# !/usr/bin/env python

from std_msgs.msg import String
import rospy

topic_rs_status = '/RS_status'

def init_node():
    global msg, pub, rate
    rospy.init_node("RS")
    pub = rospy.Publisher(topic_rs_status, String, queue_size=10)
    rate = rospy.Rate(10)
    msg = String()

def run():
    init_node()
    msg.data = 'IDLE'
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    run()