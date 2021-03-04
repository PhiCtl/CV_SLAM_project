from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyrealsense2 as rs

class Plant_holder(): # TO DO : Do we detect a single (bigger one) plant holder in one frame or several ? Is this class relevant ?
    # -> convert into an Object_detection class which can retrieve their positions, planes and so on. -> Handle several
    
    def __init__(self): # TO DO: Members to refine
        
        self.centroid = None
        self.centroid_depth = None
        self.coordinates = None
        self.normal_plane = None
        
    def get_mask(img, plot = False): #TO DO : if several objects are detected, what can we do ?
    
        """Returns mask of object contained in picture """
    
        #smoothen
        img = cv2.medianBlur(img,5)

        #resize # TO DO - MUST BE REMOVED when dealing with camera frames
        scale = 0.1
        new_height = int(img.shape[0]*scale)
        new_width  = int(img.shape[1]*scale)
        img = cv2.resize(img, (new_width, new_height),interpolation = cv2.INTER_AREA)
    
        #define BGR orange color range for BGR thresholding
        l_orange = np.array([10,50,175],dtype = np.float32)
        u_orange = np.array([40,73,210],dtype = np.float32)
    
        #opening to get rid of the small background artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    
        #thresholding and open (TO DO : tune so that only the square part can be properly detected)
        mask = cv2.inRange(img, l_orange, u_orange)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if plot:
            #bitwise and mask and original picture
            res = cv2.bitwise_and(img,img, mask= mask)
            cv2.imshow('result',res)
            cv2.imshow('mask BGR',mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return mask, img #is it necessary to return img ?
    
    
    def find_centroid(img, verbose = False): #OK
        """Returns centroid in image coordinates
            Args: 2D image (3 channels RGB)
                  mask (1/0)
        """
        
        mask, _ = self.get_mask(img)
        
        # Pick the main object and find its moments TO DO : tune threshold
        #find moments based on contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #define threshold to select bigger object
        threshold = 200  
        contours = [el for el in contours if cv2.contourArea(el) > threshold]
        cnt = contours[0]
        M = cv2.moments(cnt)

        # Find centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    
        # Plots
        if verbose:
            # Centroid pixels coordinates
            print("x : {}, y : {}".format(cx,cy))
            # Print centroid
            cv2.circle(img, (int(cx),int(cy)), 2, 255, 2)
            # Draw contours
            cv2.drawContours(img, contours, -1, (255, 0, 0), 2) # image, contours, contourIdx, color, thickness
            cv2.imshow('centroid',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        self.centroid = [cx, cy]
        
    def get_plane_orientation(mask, img, depth): #TO TEST
        """
        Computes normal vector of object plane
        Args: mask (1/0) with only one object detected at the moment
              2D image (2D vector 3 channels)
              depth (2D vector 1 channel)"""
        
        # 1- Compute object coordinates

        # compute object depth by applying a mask
        obj_depth = mask * depth
        # compute object coordinates
        xy_i = np.nonzero(obj_depth) #(x,y) pixels coordinates, corrected nan entries
        nz_objdepth = obj_depth[np.nonzero(obj_depth)] #depth (z) coordinates BUT pbm with return type -> returns tuple
        # convert in camera coordinates frame
        
        # convert in world coordinates frame
        points_abs = #([x,y,z]) dim: N x 3 absolute position

        # 2- Find 3D plane
            
        # compute 3D plane variables and normal
        pts = Points(points_abs) #must be built with a nd.array
        plane = Plane.best_fit(points)
        
        self.normal_plane = np.array(plane.normal)