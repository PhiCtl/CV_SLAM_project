# !/usr/bin/env python

"""
Test if matrix multiplication is equivalent to rs2.rs2_deproject_pixel_to_point
i.e. if distortion is negligible.

"""

import __init__
from cv_lib.src.camera_listener import CameraListener
import numpy as np

# topics list
topic_rs_status = '/RS_status'
topic_cv_status = '/CV_status'
topic_cv_data = '/CV_data'
topic_cv_obj = '/CV_object'

def test():
    camera = CameraListener()

    # Get frames and camera intrinsics
    camera.get_frames()
    camera.get_info()
    # Get intrinsics as a matrix
    K = camera.get_matrix()
    K_inv = np.linalg.inv(K)

    # Pixel arbitrary coordinates
    rd_pixel = np.array([[200, 200],
                        [100,150],
                        [150,50],
                         [300,300]]) # (4,2)

    # Pixel coordinates need to be reverted to array indexing -> (u,v) = (y,x)
    # should be (4,)
    rd_pixel_depth = np.array([camera.depth_frame[j,i] for i, j in rd_pixel]) # so j,i <- i,j
    rd_pixel_augmented = np.append(rd_pixel, np.ones((len(rd_pixel),1)), axis = 1) # (4,3)
    # With distortion
    real_points = []
    # without distortion point wise
    approx1 = []
    for pixel, pixel_au, depth in zip(rd_pixel, rd_pixel_augmented, rd_pixel_depth):
        real_points.append(camera.image_2_camera(pixel, depth))
        approx1.append(depth * K_inv @ pixel_au)

    # Without distortion
    approximated_points = camera.image_2_camera(rd_pixel, rd_pixel_depth)
    # print('Real points: ', real_points)

    # Should be equal
    print('Approximated points: ', approximated_points)
    print('Approximated points point wise: ', approx1)


if __name__ == '__main__':
    test()