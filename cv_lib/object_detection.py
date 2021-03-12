from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyrealsense2 as rs

class Object_Detection(): 
    
    def __init__(self, img): #TODO: Members to refine
        """Args: input image (BGR) from which we perform object detection"""
        
        self.mask = None #mask from which we count objects and compute their positions
        self.masks = [] #list of object wise masks
        self.frame = img
        self.kmeans = None
        self.detected_obj = [] #list of (centroids) pixels positions of objects
        self.planes = [] #list of corresponding 3D planes
        self.coo = [] #objects (world) camera coordinates
        
    def reset: #TODO: implement & send info to ROS node
        self.mask, self.frame, self.kmeans = None, None, None
        self.masks, self.detected_obj, self.planes, self.coo = [], [], [], []
        
    def k_means(self, scale = 1, nb_clust = 8, save = False): # OK
        
        """Computes kmeans on input image (useful to find color range for object detection)
           Args: scaling
                 number of clusters to compute
                 save: whether we store in /data the result"""
        
        output = self.frame.copy()
        # Resize
        new_height = int(output.shape[0]*scale)
        new_width  = int(output.shape[1]*scale)
        output = cv2.resize(output, (new_width, new_height),interpolation = cv2.INTER_AREA)
        
        # Kmeans #TODO understand better kmeans
        clust = output.reshape((-1,3))
        clust = np.float32(clust) #should be flattened and of type float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1.0) #max iter and accuracy epsilon
        K = nb_clust 
        ret, label, center = cv2.kmeans(clust, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        clust = res.reshape((output.shape))
        
        # Print
        cv2.imshow('K-means',clust)
        cv2.imshow('Original', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        if save:
            cv2.imwrite('../data/kmeans.jpg', clust)
            # Print
            cv2.imshow('K-means',clust)
            cv2.imshow('Original', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self.kmeans = clust
            
    def get_mask(self, sens = 10, it = 2, scale = 1, plot = False): #OK
    
        """Finds objects in frame
        args: sensitivity of color range
              number of iterations to erode the mask
              scaling of picture
              plot resulting mask"""
        
        # Smoothen
        output = self.frame.copy()
        output = cv2.medianBlur(output,5)
    
        # Resize
        new_height = int(output.shape[0]*scale)
        new_width  = int(output.shape[1]*scale)
        output = cv2.resize(output, (new_width, new_height),interpolation = cv2.INTER_AREA)
        
        # Define HBR orange color range for thresholding TUNED :)
        low_orange = np.array([16.8/2-sens,0.5*255,0.5*255], dtype = np.float32) 
        upp_orange = np.array([16.8/2 +sens,255,255], dtype = np.float32)
            
        # Opening to get rid of the small background artifacts -> #TODO : tune size of opening element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        
        # From bgr to hsv colorspace
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        
        # Threshold and erode
        mask = cv2.inRange(hsv, low_orange, upp_orange)
        mask = cv2.erode(mask,kernel,iterations = it)
        
        # Store mask
        self.mask = mask
    
        if plot:
            # Bitwise and mask and original picture
            res = cv2.bitwise_and(output,output, mask= mask)
            cv2.imshow('result',res)
            cv2.imshow('mask HSV',mask)
            cv2.imshow('img', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    
    def find_centroids(self, threshold = 1000, verbose = False): #OK
        """Stores centroids in image (pixels) coordinates and returns if found any
            Args: threshold for max area detection
        """
        
        output = self.frame.copy()
        mask_obj = np.zeros(output.shape[0:2]) # 2D mask 1 channel only
        
        # Pick the main objects and find its moments
        #find moments based on contours
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [el for el in contours if cv2.contourArea(el) > threshold]
        for el in contours:
            M = cv2.moments(el)

            # Find centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            self.detected_obj.append([cx,cy])
            
            # Extract contours
            self.masks.append(cv2.fillConvexPoly(mask_obj, el, color = (255,255,255) ))
    
            # Plots
            if verbose:
                # Centroid pixels coordinates
                print("x : {}, y : {}".format(cx,cy))
                # Print centroid and show object mask
                cv2.circle(output, (int(cx),int(cy)), 2, 255, 1)
                cv2.imshow('object mask',self.masks[-1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Reset mask object
            mask_obj = np.zeros(output.shape) 
            
        # Draw contours
        if verbose:
            # Draw contours
            cv2.drawContours(output, contours, -1, (255, 0, 0), 2) # image, contours, contourIdx, color, thickness
            cv2.imshow('centroid',output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return not self.detected_obj #check if objects centroid were found
    
    def get_pos(self, camera): #OK #TODO: get centroid positions or get points position ?
    
        """Computes positions of each object in camera 3D coordinates frame
           Args: camera from which we retrieve the depth and the intrinsics"""
        
        # Retrieve depth from camera
        depth_frame = camera.frames.get_depth_frame()
        # Compute depth of each centroid
        for [cx,cy] in self.detected_obj:
            cz = depth_frame.get_distance(cx,cy)
            # Store each object in camera coordinates
            pos = camera.image_2_camera([cx,cy], cz)
            print("Coordinates : {}".format(pos))
            self.coo.append(pos)
        
    def get_plane_orientation(self, camera, plot = False ): #TODO : Correct get plane orientation and print arrow
        """
        Computes normal vector of object plane
        Args: RS_camera object
        """
        
        thresh = 1e-6 #TODO: is it useful ?
        
        # Retrieve depth from camera
        depth_frame = camera.frames.get_depth_frame()
        
        # For each region, compute plane
        for mask in self.masks:
    
            # Compute 3D coordinates of all pixels within the object
            obj_pix = np.argwhere(mask > thresh)
            points = []
            for p in obj_pix: #TODO: avoid loop here !
                z = depth_frame.get_distance(p[0],p[1])
                points.append(camera.image_2_camera([p[0],p[1]], z) )
            
            # Feed into Points, best fit etc... 
            pts = Points(np.array(points)) #must be built with a nd.array
            plane = Plane.best_fit(pts)
            if plot:
                print("Normal to plane : {}".format(np.array(plane.normal)))
        
            # Append computed plane
            self.planes.append(np.array(plane.normal))
        