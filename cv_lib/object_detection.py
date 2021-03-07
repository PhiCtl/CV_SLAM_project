from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyrealsense2 as rs

class Object_detection(): # TO DO : Do we detect a single (bigger one) plant holder in one frame or several ?
    
    def __init__(self): # TO DO: Members to refine
        
        self.mask = None #mask from which we count objects and compute their positions
        self.detected_obj = [] #list of (centroids) pixels positions of objects
        self.planes = [] #list of corresponding 3D planes
        self.world_coo = [] #objects (world) camera coordinates
        
    def update(publisher): # TO DO: send info to SLAM 3D map ?
        self.mask = None
        # store detected objects coordinates and planes somewhere
        # erase everything
       
    def get_mask(img, plot = False): #TO DO: test with several light intensities and several objects
    
        """Stores mask of object(s) contained in picture
           Args: 3 channels RGB frame (from camera)"""
    
        #smoothen
        img = cv2.medianBlur(img,5)

        #resize # TO DO - MUST BE REMOVED when dealing with camera frames
        scale = 0.1
        new_height = int(img.shape[0]*scale)
        new_width  = int(img.shape[1]*scale)
        img = cv2.resize(img, (new_width, new_height),interpolation = cv2.INTER_AREA)
    
        #define BGR orange color range for BGR thresholding -> TO DO: to be experimentally tuned (and camera calibration)
        l_orange = np.array([10,50,175],dtype = np.float32)
        u_orange = np.array([40,73,210],dtype = np.float32)
    
        #opening to get rid of the small background artifacts.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    
        #thresholding and open (TO DO : tune so that only the square part can be properly detected)
        mask = cv2.inRange(img, l_orange, u_orange)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if plot:
            #bitwise_and mask and original picture
            res = cv2.bitwise_and(img,img, mask= mask)
            cv2.imshow('result',res)
            cv2.imshow('mask BGR',mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        #detect objects based on contour area
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #define threshold to select objects -> TODO : experimentally ?
        threshold = 200  
        contours = [el for el in contours if cv2.contourArea(el) > threshold]
        mask_objects = np.zeros(img.shape[0:2], np.uint8)
        #select only relevant contours on final mask
        cv2.drawContours(mask_objects, contours, -1, color=1, thickness=cv2.FILLED)
        
        self.mask = mask_objects
    
    
    def find_centroids(img, verbose = False): #TO DO: test with several objects
        """Stores centroids in image (pixels) coordinates and returns if found any
            Args: 2D image (3 channels RGB) from camera
        """
        
        # Pick the main objects and find its moments
        #find moments based on contours
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for el in contours:
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
            self.detected_obj.append([cx,cy])
         
        return not self.detected_obj #check if objects centroid were found
    
    def get_pos(camera): #TODO : TEST
        
        #retrieve depth from camera
        depth_frame = camera.frames.get_depth_frame()
        #compute depth of each centroid
        for [cx,cy] in self.detected_obj:
            cz = depth_frame.get_distance(cx,cy)
            #store each object in camera coordinates
            pos = camera.image_2_camera([cx,cy], cz)
            self.world_coo.append()
        
    def get_plane_orientation(camera): #TODO : TEST
        """
        Computes normal vector of object plane
        Args: camera from which we can retrieve the depth and convert pixels to world coordinates
        """
        
        # 1- Compute object coordinates
        # convert in camera coordinates frame
        
        # convert in world coordinates frame TODO: find extrinsic parameters
        #points_abs = #([x,y,z]) dim: N x 3 absolute position

        # 2- Find 3D plane
            
        # compute 3D plane variables and normal
        pts = Points(points_abs) #must be built with a nd.array
        plane = Plane.best_fit(points)
        
        self.normal_plane = np.array(plane.normal)