import pyrealsense2 as rs2
import cv2
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# topics list
topic_camera_rgb = '/camera/color/image_raw'
topic_camera_depth = '/camera/aligned_depth_to_color/image_raw'
topic_camera_info = '/camera/aligned_depth_to_color/camera_info'
# TODO difference between rospy.wait_message and rospy.Subsriber(...)


########################################################################################################################
#        Using scripts from
#        https://github.com/IntelRealSense/realsense-ros/blob/development/realsense2_camera/scripts/show_center_depth.py
########################################################################################################################

class Camera_Listener(): # TODO test

    def __init__(self):

        # suscriptions #TODO can I perform the subscribers initialization into another method ?
        self.sub_depth = None
        self.sub_info = None
        self.sub_rgb = None

        # frames
        # self.frames = None
        self.bgr_image = None
        self.depth_frame = None
        self.bridge = CvBridge()

        # camera intrinsics
        self.intrinsics = None

        self.get_frames()
        pass

    def imageDepthCallback(self, data):
        """Args: Image from sensor_msgs.msg
        from github repo specified above"""
        try:
            # convert to CV image depth
            self.depth_frame = self.bridge.imgmsg_to_cv2(data, data.encoding)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageInfoCallback(self, cameraInfo):
        """Args: CameraInfor from sensor_msgs.msg
           from github repo specified above"""
        try:
            if self.intrinsics:  # if already filled in
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

    def imageRGBCallback(self, data):
        """Args: Image from sensor_msgs.msg
        from github repo specified above"""
        try:
            # convert to CV BGR image
            self.bgr_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")  # BGR format

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def get_frames(self):
        self.sub_depth = rospy.Subscriber(topic_camera_depth, Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(topic_camera_info, CameraInfo, self.imageInfoCallback)
        self.sub_rgb = rospy.Subscriber(topic_camera_rgb, Image, self.imageRGBCallback)

    def get_distance(self, cx, cy):  # TODO beware of '0' situation
        return self.depth_frame[cy, cx]  # y,x convention in opencv

    def image_2_camera(self, pixels):
        [cx, cy] = pixels
        if self.intrinsics:
            depth = self.get_distance(cx, cy)
            return rs2.rs2_deproject_pixel_to_point(self.intrinsics, pixels, depth)
        else:  # TODO error handling here
            pass
