from skspatial.objects import Points, Plane
#from skspatial.plotting import plot_3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs

"""
Object detection by HSV thresholding
"""

class ObjectDetector():

    def __init__(self):
        """Args: input image (BGR) from which we perform object detection"""
        
        self.mask = None #mask from which we count objects and compute their positions
        self.masks = [] #list of object wise masks
        self.frame = None
        self.kmeans = None
        self.detected_obj = [] #list of (centroids) pixels positions of objects
        self.planes = [] #list of corresponding 3D planes
        self.coo = [] #objects (world) camera coordinates
        
    def reset(self):
        self.mask, self.frame, self.kmeans = None, None, None
        self.masks, self.detected_obj, self.planes, self.coo = [], [], [], []

    def set_picture(self, img):
        self.frame = img
        
    def k_means(self, scale = 1, nb_clust = 8, save = False): # OK
        
        """Computes kmeans on input image (useful to find color range for object detection)
           Args: scaling
                 number of clusters to compute
                 save: whether we store in /data the result"""
        
        output = self.frame.copy()
        # Resize
        #new_height = int(output.shape[0]*scale)
        #new_width  = int(output.shape[1]*scale)
        #output = cv2.resize(output, (new_width, new_height),interpolation = cv2.INTER_AREA)
        
        # Kmeans
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
            cv2.imwrite('../data/plant_holder/kmeans.jpg', clust)
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

        if plot : print("Getting mask...")
        # Smoothen
        output = self.frame.copy()
        output = cv2.medianBlur(output,5)
    
        # Resize
        #new_height = int(output.shape[0]*scale)
        #new_width  = int(output.shape[1]*scale)
        #output = cv2.resize(output, (new_width, new_height),interpolation = cv2.INTER_AREA)
        
        # Define HBR orange color range for thresholding TUNED :)
        low_orange = np.array([16.8/2-sens,0.5*255,0.5*255], dtype = np.float32) 
        upp_orange = np.array([16.8/2 +sens,255,255], dtype = np.float32)
            
        # Opening to get rid of the small background artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        
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
            print("Closing windows...")
            print("Done")
    
    
    def find_centroids(self, threshold = 2000, verbose = False): #OK
        """Stores centroids in image (pixels) coordinates and returns if found any
            Args: threshold for max area detection
        """
        if verbose: print("Finding centroids...")
        output = self.frame.copy()
        
        # Pick the main objects and find its moments
        #find moments based on contours
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [el for el in contours if cv2.contourArea(el) > threshold]
        for el in contours:
            M = cv2.moments(el)

            # Find centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            self.detected_obj.append([cx,cy]) # pixel coordinates (u,v), correct Open CV convention order
            
            # Extract contours
            mask_obj = np.zeros((output.shape[0], output.shape[1])) # 2D mask 1 channel only
            self.masks.append(cv2.fillConvexPoly(mask_obj, el, color = (255,255,255) ))
            #cv2.drawContours(mask_obj, el, -1, (255, 255, 255), 2) # image, contours, contourIdx, color, thickness
            #self.masks.append(mask_obj)
            # Centroid pixels coordinates
            #print("x : {}, y : {}".format(cx, cy))
            
            # Plots
            if verbose:
                # Print centroid and show object mask
                cv2.circle(output, (int(cx),int(cy)), 2, 255, 1)
                cv2.imshow('object mask',mask_obj)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        # Draw contours
        if verbose:
            # Draw contours
            cv2.drawContours(output, contours, -1, (255, 0, 0), 2) # image, contours, contourIdx, color, thickness
            cv2.imshow('centroid',output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Done")
        return len(self.detected_obj) > 0 #check if objects centroid were found
    
    def get_pos(self, camera, verbose = False): #OK
    
        """Computes positions of each object in camera 3D coordinates frame
           Args: camera from which we retrieve the depth and the intrinsics"""
        
        # Retrieve depth from camera
        # Compute depth of each centroid (pixel coordinate)
        if verbose: print("Finding 3D coordinates...")
        for [cx,cy] in self.detected_obj:
            cz = camera.get_distance(cx,cy)
            # Store each object in camera coordinates
            pos = camera.image_2_camera((cx,cy), cz)
            [x,y,z] = pos
            print("Coordinates : {}".format(pos))
            self.coo.append(np.array([x/1000, y/1000, z/1000]))
        if verbose: print("Done")
        
    def get_plane_orientation(self, camera, plot = False ): 

        """
        Computes normal of object plane and plot
        Args: RS_camera object
        """
        
        thresh = 1e-6
        # Set plot
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        
        # For each region, compute plane
        for mask in self.masks:
    
            # Compute 3D coordinates of all pixels within the object
            obj_indexes= np.argwhere(mask > thresh) # array indexing (x,y)
            obj_pix = np.empty((obj_indexes.shape[0],2))
            # needs to be reverted here -> to pixel coordinates (u,v) = (y,x)
            obj_pix[:,1] = obj_indexes[:,0]
            obj_pix[:,0] = obj_indexes[:,1]
            points = camera.image_2_camera(obj_pix, camera.depth_frame[obj_indexes[:,0], obj_indexes[:,1]])

            # Feed into Points, best fit etc...
            pts = Points(points) #must be built with a nd.array
            plane = Plane.best_fit(pts)
            # Append computed plane
            self.planes.append(np.array(plane.vector))
            
            # Plot
            if plot:
                print("Plane vector : {}".format(np.array(plane.vector)))
                # Plot plane
                [X,Y,Z] = self.coo[len(self.planes)-1]
                [U,V,W] = np.array(plane.vector) #self.planes[-1]
                #[x,y,z] = [points[:,0],points[:,1],points[:,2]] 
                ax.quiver(X, Y, Z, U, V, W)
                #ax.scatter(x, y, z, c='r', marker='o')
                
                
        if plot:
            print("Plotting...")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-2, 2])
            ax.set_xlabel('x in m')
            ax.set_ylabel('y in m')
            ax.set_zlabel('z in m')
            plt.show()
            print("Done")
            
        
        
