# !/usr/bin/env python

"""
Main ros node for object detection and localization
Currently only plant holders can be detected with this framework

---------------------------------------------------------------------
Parameters that can be changed (based on custom tuning):
- in detector.get_mask (l.172) :
    - hsv_low threshold : [0,0,0] <= [h, s, v] <= [179, 255, 255]
    - hsv_high threshold : [0,0,0] <= [h, s, v] <= [179, 255, 255]
    - k_size kernel size for erosion : (int)
    - it number of iterations for erosion : (int)
- in detector.find_centroids (l. 173) :
    - threshold: (int > 1000) for detection of object with area above threshold
    should be of about 500 for robotic arm scan position R2 and about 2'000 otherwise
- average accuracy n (l. 159) should not be larger than 6, otherwise too much delay
"""

import __init__
from cv_lib.src.object_detection import ObjectDetector
from cv_lib.src.camera_listener import CameraListener
from cv_lib.src.object_prediction import ObjectPredictor
from cv_lib.src.tracker import Tracker
from cv_lib.src.myUtils import draw_bboxes
from cv_ros.msg import ObjectPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import tf2_ros, rospy, torch, cv2, quaternion
import numpy as np

# topics list
topic_rs_status = '/RS_status'
topic_cv_status = '/CV_status'
topic_cv_data = '/CV_data'
topic_cv_obj = '/CV_object'

class PosePublisher():

    def __init__(self):

        # Init node
        rospy.loginfo("init_node")
        rospy.init_node("Vision")

        # Init publisher
        self.pub = rospy.Publisher(topic_cv_data, ObjectPose, queue_size=10)
        self.pub_state = rospy.Publisher(topic_cv_status, String, queue_size=10)
        self.pub_obj = rospy.Publisher(topic_cv_obj, String, queue_size=10)

        # Msg, buffer for camera to world transfo, status
        self.msg = ObjectPose()
        self.cv_status = String()
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.rate = rospy.Rate(10)

    def publish_pose(self, centroid_coo, plane_vector_coo, seq_id, object_type):
        """
        Publishes per camera frame ObjectPose message containing poses of detected objects
        :param centroid_coo: list of centroids coordinates -> total dim : (nb_detections x 3 coordinates)
        :param plane_vector_coo: list of planes -> total dim : (nb_detections x 3 coordinates)
        :param seq_id: msg id
        :param object_type: (string) type of detected object

        """
        self.msg.header.frame_id, self.msg.header.stamp, self.msg.header.seq, self.msg.object_type = "world", rospy.Time.now(), seq_id, String(object_type)

        for centroid, plane, i in zip(centroid_coo, plane_vector_coo, range(len(centroid_coo))):

            p = PoseStamped()
            p.header.frame_id, p.header.seq = 'camera', i

            # Position in camera coordinates
            p.pose.position.x = centroid[0]
            p.pose.position.y = centroid[1]
            p.pose.position.z = centroid[2]

            # Orientation in camera coordinates (plane vector)
            [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z] = plane

            # Change orientation and position to world coordinates, compute angles with world frame X, Y and Z
            #p = self.transform_cam_to_world_coo_cb(p) # TODO uncomment
            self.msg.poses.append(p)

        self.pub.publish(self.msg)

        # Update msg index
        seq_id += 1

        # Empty poses array
        self.msg.poses.clear()

    def publish_status(self):
        # publish status
        self.pub_state.publish(self.cv_status)

    def publish_obj_detected(self, data):
        # publish number and type of detected object(s)
        self.pub_obj.publish(data)

    def set_status(self, str='IDLE'):
        # set status
        self.cv_status.data = str

    def transform_cam_to_world_coo_cb(self, data):
        """
        Transforms camera to world coordinates
        param: data (ObjectPose message) with coordinates and orientations in camera coordinates
        return. data 8ObjectPose message) with coordinates and orientations in world coordinates
        """

        # Look for transform
        trans = self.buffer.lookup_transform("world", "camera", rospy.Time(0) )
        data.header.frame_id = "world"

        # retrieve rotations
        rot = quaternion.quaternion()
        rot.x = trans.transform.rotation.x
        rot.y = trans.transform.rotation.y
        rot.z = trans.transform.rotation.z
        rot.w = trans.transform.rotation.w

        # Convert positions to world coordinates
        pos = quaternion.from_float_array([0, data.pose.position.x, data.pose.position.y, data.pose.position.z])
        pos = rot * pos * rot.inverse()
        [data.pose.position.x, data.pose.position.y, data.pose.position.z] = quaternion.as_vector_part(pos)

        # Convert orientations to world coordinates
        normal_vec = quaternion.from_float_array([0, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
        normal_vec = rot * normal_vec * rot.inverse()
        [x, y, z] = quaternion.as_vector_part(normal_vec)

        # Compute angles with X, Y, Z world ref vectors
        norm = np.sqrt(x**2 + y**2 + z**2)
        data.pose.orientation.x = np.arccos(x / norm)
        data.pose.orientation.y = np.arccos(y / norm)
        data.pose.orientation.z = np.arccos(z / norm)

        # Add translation to position in world coordinates
        data.pose.position.x += trans.transform.translation.x
        data.pose.position.y += trans.transform.translation.y
        data.pose.position.z += trans.transform.translation.z

        return data

def run():
    publisher = PosePublisher()

    # # start camera listener and detector
    camera = CameraListener()
    detector = ObjectDetector()
    predictor = ObjectPredictor()
    tracker = Tracker(0.001)
    publisher.set_status()

    # counters
    i, j = 1,1
    n = 1

    # utils
    rs_status = String('ELSE') # to refine

    print('Ready')

    while not rospy.is_shutdown():

        # # Wait for Robotic station status
        try:
            rs_status = rospy.wait_for_message(topic_rs_status, String, timeout=1)
        except(rospy.exceptions.ROSException):
            # If we were already in the STANDSTILL setting, we stay in standstill
            # Othwerwise, we do nothing
            if rs_status.data == 'STANDSTILL':
                pass

        if rs_status.data == 'STANDSTILL':

            # Take picture
            camera.get_frames()
            camera.get_info()
            cv2.imwrite('output.jpg', camera.bgr_image)

            # detect plant_holder
            detector.set_picture(camera.bgr_image)
            detector.get_mask(it=2)
            found_plantHolders = detector.find_centroids(threshold=1000, verbose=True)

            # Find positions and orientations
            detector.get_pos(camera, verbose=False)
            detector.get_plane_orientation(camera, plot=False)

            # detect flowers
            poses, found_flowers = predictor(camera.bgr_image, camera, conf = 0.3, verbose=False)

            #If we found any plant holder
            if found_plantHolders:

                tracker.update(detector.coo, detector.planes, 'plant_holder')
                # Make mean over n values
                if i%n == 0:
                    obj_list, type = tracker.get_object_list(type='plant_holder')
                    publisher.publish_pose(obj_list[0], obj_list[1], i, 'plant_holders')
                # This is a success
                publisher.set_status('SUCCESS')

                # Return number and type of detected objects
                msg = str(tracker.nb_detected['plant_holder']) + ' plant holder(s) detected'
                publisher.publish_obj_detected(msg)

                # Reset detector and update counter
                detector.reset()
                i += 1

            if found_flowers:

                tracker.update(poses['centroids_coo'][:,:3], poses['orientations'], 'flower')

                if j%n == 0:
                    obj_list, type = tracker.get_object_list('flower')
                    publisher.publish_pose(obj_list[0], obj_list[1], j, 'flower')
                # Update counter
                j += 1

                # Publish state and number of flower detected
                publisher.set_status('SUCCESS')
                msg = str(poses['nb_detected']) + ' flower(s) detected'
                publisher.publish_obj_detected(msg)

            if not found_flowers and not found_plantHolders:
                publisher.set_status('RETRY')
                publisher.publish_obj_detected('None')
        publisher.publish_status()
        publisher.rate.sleep()


if __name__ == '__main__':
    run()
    # test()