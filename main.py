# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:50:29 2021

@author: Philippine
"""
import os, cv2, torch
import numpy as np
from cv_lib.src.rs_camera import RS_Camera
from cv_lib.src.object_detection import ObjectDetector
from cv_lib.src.object_prediction import ObjectPredictor

currentPath = os.path.dirname(os.path.realpath(__file__))

def run():

    # Start camera and get frames
    camera = RS_Camera()
    camera.start_RS()
    camera.get_frames()

    #predict flowers
    predictor = ObjectPredictor(model_name='YOLOv5x')
    poses, found_flowers = predictor(camera.bgr_image, camera, conf=0.4, verbose=True)
    if found_flowers: print(poses)

    
if __name__ == '__main__':
    run()
