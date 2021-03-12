# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:50:29 2021

@author: Philippine
"""
import os
import numpy as np
import cv2
from cv_lib.RS_Camera import RS_Camera
from cv_lib.Object_Detection import Object_Detection

currentPath = os.path.dirname(os.path.realpath(__file__))

def run():
    
    # Start camera and get frames
    camera = RS_Camera()
    camera.start_RS()
    camera.get_frames()
    
    #detect object
    detector = Object_Detection(camera.bgr_image)
    #detector.k_means(save = True)
    detector.get_mask()
    detector.find_centroids(threshold = 100)
    detector.get_plane_orientation(camera, plot = True)
    
    
if __name__ == '__main__':
    run()