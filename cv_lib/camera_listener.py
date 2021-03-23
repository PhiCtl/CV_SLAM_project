import pyrealsense2 as rs2
import cv2
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# topics list
topic_camera_rgb = '/camera/color/image_raw'
topic_camera_depth = '/camera/aligned_depth_to_color/image_raw'
topic_camera_info = '/camera/aligned_depth_to_color/camera_info'


########################################################################################################################
#        Using scripts from
#        https://github.com/IntelRealSense/realsense-ros/blob/development/realsense2_camera/scripts/show_center_depth.py
########################################################################################################################

class CameraListener: # TODO test

    def __init__(self):

        # frames
        self.bgr_image = None
        self.depth_frame = None
        self.bridge = CvBridge()

        # camera intrinsics
        self.intrinsics = None

    def get_frames(self):
        img = rospy.wait_for_message(topic_camera_rgb, Image)
        img_depth = rospy.wait_for_message(topic_camera_depth, Image)
        try:
            # convert to CV BGR image
            self.bgr_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")  # BGR format
            self.depth_frame = self.bridge.imgmsg_to_cv2(img_depth, img_depth.encoding)
        except CvBridgeError as e:
            print(e)
        except ValueError as e:
            print(e)

    def get_info(self):
        cameraInfo = rospy.wait_for_message(topic_camera_info, CameraInfo)
        try:
            if not self.intrinsics:
                self.intrinsics = rs2.intrinsics()
                self.intrinsics.width = cameraInfo.width
                self.intrinsics.height = cameraInfo.height
                self.intrinsics.ppx = cameraInfo.K[2]
                self.intrinsics.ppy = cameraInfo.K[5]
                self.intrinsics.fx = cameraInfo.K[0]
                self.intrinsics.fy = cameraInfo.K[4]
                if cameraInfo.distortion_model == 'plumb_bob':
                    self.intrinsics.model = rs2.distortion.brown_conrady
                if cameraInfo.distortion_model == 'equidistant':
                    self.intrinsics.model = rs2.distortion.kannala_brandt4
                self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)


    def get_distance(self, cx, cy):  # TODO beware of '0' situation
        return self.depth_frame[cy, cx]  # y,x convention in opencv

    def image_2_camera(self, pixels, depth):
        [cx, cy] = pixels
        if self.intrinsics:
            return rs2.rs2_deproject_pixel_to_point(self.intrinsics, pixels, depth)
        else:  # TODO error handling here
            pass
