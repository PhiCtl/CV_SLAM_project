#######################################################################################################
                                # from ThibaultNiederhauser repository                                #
# https://github.com/ThibaultNiederhauser/potato_harvesting_vision/blob/master/vision_lib/rs_camera.py#
#######################################################################################################

import pyrealsense2 as rs
import cv2
import numpy as np

class RS_Camera():

    def __init__(self):

        self.pipeline = None
        self.colorizer = None
        self.depth_scale = None
        #self.start_RS()

        self.frames = None
        self.bgr_image = None
        self.depth_frame = None
        self.colorized_depth = None
        self.h = 0
        self.w = 0


    def start_RS(self):
        # Create pipeline : Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()
        # Configure streams
        config = rs.config()
        # Configure the pipeline to stream the depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # read in BRG because of opencv
        # Start streaming from file
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        self.pipeline = pipeline
        self.depth_scale = depth_sensor.get_depth_scale()
        self.colorizer = rs.colorizer()

    def get_frames(self):
        # Get frameset of depth
        frames = self.pipeline.wait_for_frames()
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        # Update color and depth frames:
        self.depth_frame = frames.get_depth_frame()
        colorized_depth = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())
        # Get depth frame
        bgr_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        colorized_depth = self.colorizer.colorize(self.depth_frame)
        # Convert depth_frame to numpy array to render image in opencv
        colorized_depth = np.asanyarray(colorized_depth.get_data())
        bgr_image = np.asanyarray(bgr_frame.get_data())

        self.bgr_image = bgr_image
        self.colorized_depth = colorized_depth #TODO : understand what colorized depth is and its advantages
        self.frames = frames
        self.h, self.w = bgr_image.shape[:2]
    
    def image_2_camera(self, points, depth): #TODO: modify for arrays and use matrix product
        # Make use of https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L46
        """Converts object 2D coordinates to 3D camera space coordinates
        INTRINSIC PARAMETERS
        Args: points tuple
              depth  single value
              """
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        
        pos = rs.rs2_deproject_pixel_to_point(depth_intrinsics, points, depth)
        
        return pos

    def get_distance(self, cx, cy):
        depth_frame = self.frames.get_depth_frame()
        return depth_frame.get_distance(cx, cy)

    
class Camera:

    def __init__(self, intrinsics):
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = intrinsics['width']
        self.intrinsics.height = intrinsics['height']
        self.intrinsics.ppx = intrinsics['p'][0]
        self.intrinsics.ppy = intrinsics['p'][1]
        self.intrinsics.fx = intrinsics['f'][0]
        self.intrinsics.fy = intrinsics['f'][1]
        self.intrinsics.model = rs.distortion.brown_conrady

        self.intrinsics.coeffs = [i for i in intrinsics['D']]

    def set_frames(self, bgr_pic, depth_pic):
        self.bgr_image = bgr_pic
        self.depth_frame = depth_pic

    def image_2_camera(self, pixels, depth):
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, pixels, depth)

    def get_distance(self, cx, cy):
        z = self.depth_frame[cy, cx]
        return z  # y,x convention in opencv
