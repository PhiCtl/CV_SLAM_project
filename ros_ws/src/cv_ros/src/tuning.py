# !/usr/bin/env python

"""
Additional node for HSV-thresholding-based detection tuning
---------------------------------------------------------
Several parameters can be tuned:
-> HSV lower and upper range
-> erosion kernel size
-> erosion iterations
-> threshold for contours detection

----------------------------------------------------------
How to tune:
1) launch node
2) tune all above parameters such that
    - centroids are correctly set
    - the number of detected objects matches with the real number of detected object
3) write down parameters and modify if needed in main_ros.py

----------------------------------------------------------
It should already be tuned for current application

"""

import __init__
import cv2
import numpy as np
from cv_lib.camera_listener import CameraListener
import rospy


def nothing(x):
    pass

def find_centroids(mask, threshold=2000):  # OK
    """Finds centroids in image (pixels) coordinates and print on image
    """
    output = mask.copy()

    # Pick the main objects and find its moments
    # find moments based on contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [el for el in contours if cv2.contourArea(el) > threshold]
    for el in contours:
        M = cv2.moments(el)

        # Find centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Print centroid and show object mask
        cv2.circle(output, (int(cx), int(cy)), 2, 0, 1)

    return output, len(contours)

def create_hsv_trackbar(test_img, scale = 1):
    """
    Creates a GUI to compute the right HSV range for efficient detection
    Window closes when escape key is pressed
    :param test_img: color frame from camera
    :param scale: to rescale the image (opt)
    :return:
    """
    new_height = int(test_img.shape[0] * scale)
    new_width = int(test_img.shape[1] * scale)
    img = cv2.resize(test_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = hsv.copy()

    cv2.namedWindow('Mask_tuning')

    # create trackbars for color change
    cv2.createTrackbar('HL','Mask_tuning',0,180,nothing)
    cv2.createTrackbar('HH', 'Mask_tuning', 0, 180, nothing)
    cv2.createTrackbar('SL','Mask_tuning',0,255,nothing)
    cv2.createTrackbar('SH', 'Mask_tuning', 0, 255, nothing)
    cv2.createTrackbar('VL','Mask_tuning',0,255,nothing)
    cv2.createTrackbar('VH', 'Mask_tuning', 0, 255, nothing)
    # kernel size
    cv2.createTrackbar('s', 'Mask_tuning', 3, 11, nothing)
    # nb of iterations
    cv2.createTrackbar('it', 'Mask_tuning',1, 4, nothing)
    # threshold
    cv2.createTrackbar('th', 'Mask_tuning', 100, 3000, nothing)

# create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Mask_tuning',0,1,nothing)

    font = cv2.FONT_HERSHEY_DUPLEX
    while(1):
        cv2.imshow('Vision', img)
        cv2.imshow('Mask_tuning', mask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of trackbars
        hl = cv2.getTrackbarPos('HL','Mask_tuning')
        hh = cv2.getTrackbarPos('HH', 'Mask_tuning')
        sl = cv2.getTrackbarPos('SL','Mask_tuning')
        sh = cv2.getTrackbarPos('SH','Mask_tuning')
        vl = cv2.getTrackbarPos('VL','Mask_tuning')
        vh = cv2.getTrackbarPos('VH', 'Mask_tuning')
        sw = cv2.getTrackbarPos(switch,'Mask_tuning')
        s = cv2.getTrackbarPos('s', 'Mask_tuning')
        it = cv2.getTrackbarPos('it', 'Mask_tuning')
        t = cv2.getTrackbarPos('th', 'Mask_tuning')


        if sw == 0:
            mask[:] = 0
        else:
            low = np.array([hl, sl, vl], dtype=np.float32)
            upp = np.array([hh, sh, vh], dtype=np.float32)
            mask = cv2.inRange(hsv, low, upp)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
            mask = cv2.erode(mask, kernel, iterations=it)
            mask, nb_detected = find_centroids(mask, threshold=t)
            cv2.putText(mask, "{} object(s) were detected.".format(nb_detected), (10,10),
                         font, 0.5, 255, 1, cv2.LINE_AA)
    cv2.destroyAllWindows()


def tune():
    rospy.init_node("Vision_tuning")
    rate = rospy.Rate(10)
    camera = CameraListener()

    while not rospy.is_shutdown():

        # Take picture
        camera.get_frames()
        camera.get_info()

        # Create trackbar for tuning
        create_hsv_trackbar(camera.bgr_image)

        rate.sleep()

if __name__ == '__main__':
    tune()
