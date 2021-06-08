from myUtils import*
import time
from skspatial.objects import Points, Plane

"""
Object detection with neural networks
"""

# Constants for strawberry flower detection by HSV thresholding
LOW = np.array([24,0,245], dtype=np.float32)
UPP = np.array([92,21,255], dtype=np.float32)



class ObjectPredictor():
    def __init__(self, model_name = 'MobileNetV3_largeFPN', weights_path = '/home/phil/Documents/Projects/CV_SLAM_project/cv_lib/models/'):
        """
        Constructor
        :param model_name: name of the pre trained model to load
        :param weights_path: from where to load the trained weights
        """

        assert model_name in ['Resnet50_FPN',
                              'MobileNetV3_largeFPN',
                              'MobileNetV3_largeFPN_320',
                              'MaskRCNN',
                              'YOLOv5x'],\
            "Error: model not found"

        # Load model
        weights_path += 'best_' + model_name + ".pt"
        self.model = get_object_detection_model(num_classes=2, mtype= model_name, weights_path= weights_path, \
                                                device=torch.device('cpu'))
        self.model_name = model_name

        # Transform
        self.transform = get_img_transformed(train=False)


    def __call__(self, image, camera, conf=0.5, name='', verbose=False):
        """
        Computes poses of detected flowers in the input image

        :param image: (np.ndarray) RGB image on which we want to detect the flowers
        :param camera: (CameraListener or RSCamera object) needed to access to depth info
        :param conf: the confidence threshold below which detections are not considered
        :param name: (string) to save prediction picture

        :return: poses, a dictionary of fields:
            - 'centroids_coo':  flowers center coordinates and score (np.ndarray of dim (N x 4) 3 coordinates and 1 score)
            - 'nb_detected' : number of flowers detected (int)
            - 'orientation': (np.ndarray of dim(N,3)) plane vectors of flowers corolla orientation
        """
        assert isinstance(image, np.ndarray), "Error: img should be of type np.ndarray"

        # Get image and convert it to handled type
        if self.model_name != 'YOLOv5x':
            img = self.transform(image).unsqueeze(0).to(torch.device('cpu')) # convert to tensor of size [1, 3, H, w]
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make prediction
        self.model.eval()
        t1 = time.time()
        with torch.no_grad():
            pred = self.model(img)
        if self.model_name == 'YOLOv5x':
            out = self.model(img)
            pred = [{'boxes': out.xyxy[0][:,:4], 'scores': out.xyxy[0][:,4].flatten(), 'labels': out.xyxy[0][:,5].flatten() + 1}]
        if verbose:
            print("Elapsed time network {} s".format(time.time() - t1))
            print("{} flowers detected.".format(len(pred[0]['scores'])))
            draw_bboxes(pred[0], image, conf=conf, name=name)

        poses = pred_to_pose(pred[0], image, camera, conf=conf, verbose=verbose)

        if verbose: print("Elapsed time overall {} s".format(time.time() - t1))


        return poses


########################################################################################################################
# Functions
########################################################################################################################

def get_mask(img, b, score, low_orange = LOW, upp_orange = UPP, it=0, conf=0.3, verbose=False):
    """
    Computes the object mask in the input image by HSV based detection

    :param img: image from which we want to detect a white flower, ndarray of size (3, H, W)
    :param b : bounding box of interest : torch.tensor(xm, ym, xM, yM)
    :param score: score of the detection
    :param low_orange: lower HSV threshold for mask detection
    :param upp_orange: upper HSV threshold for mask detection
    :param it: number of iterations for mask erosion
    :param conf: confidence threshold above which the mask is printed (if verbose is set to True)
    :return: get mask of white elements in the image
    """
    b = b.numpy().astype(int)
    cropped = crop(img, b)
    output = np.zeros((img.shape[0], img.shape[1]))
    cropped = cv2.medianBlur(cropped, 5)

    # Opening to get rid of the small background artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # From bgr to hsv colorspace
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Threshold and erode
    mask = cv2.inRange(hsv, low_orange, upp_orange)
    mask = cv2.erode(mask, kernel, iterations=it)

    if verbose and score >= conf :
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    output[b[1]:b[3], b[0]:b[2]] = mask
    return output

def crop(img, bbox):
    """
    Crops the input image in the dimensions specified in bbox
    :param img: ndarray of size (3, H, W)
    :param bbox: ndarray [xm, ym, xM, yM] : are specified top left and bottom right corners
    :return: cropped picture
    """
    [xm, ym, xM, yM] = bbox

    return img[ym:yM, xm:xM,:]

def find_centroid(mask, threshold=100, verbose=True):
    """
    By method of moments
    :param mask:
    :param threshold:
    :param verbose:
    :return:
    """

    if verbose: print("Finding centroids...")
    output = mask.copy().astype(np.uint8)

    # Pick the main objects and find its moments
    # find moments based on contours
    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnts[-1])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if verbose:
        # Print centroid and show object mask
        cv2.circle(output, (int(cx), int(cy)), 2, 255, 1)
        cv2.imshow('object mask', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Draw contours
    if verbose:
        # Draw contours
        cv2.drawContours(output, [cnts[-1]], -1, (255, 0, 0), 2)  # image, contours, contourIdx, color, thickness
        cv2.imshow('centroid', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Done")

    return cx, cy

def get_pose_and_orientation(mask, camera, verbose=False):
    """
    Computes object pose and orientation
    :param mask: object mask from which we'll retrieve the location and the orientation
    :param camera: CameraListener or RSCamera object
    :return: - x, y, z : object coordinates in mm in camera coordinates
             - vector: (np.ndarray) normal to the object plane (the object is considered as being almost flat)

    """
    # Find where is the flower in the mask
    obj_indexes= np.argwhere(mask > 1e-6)

    if (obj_indexes.size > 0):
        # We compute the orientation
        obj_pix = np.empty((obj_indexes.shape[0], 2))
        # needs to be reverted here -> to pixel coordinates (u,v) = (y,x)
        obj_pix[:, 1] = obj_indexes[:, 0]
        obj_pix[:, 0] = obj_indexes[:, 1]
        points = camera.image_2_camera(obj_pix, camera.depth_frame[obj_indexes[:, 0], obj_indexes[:, 1]])
        # We fit the plane
        pts = Points(points)  # must be built with a nd.array
        plane = Plane.best_fit(pts)

        # We compute the position
        # We compute the flower (x,y,z) location in pixels coordinates
        cx, cy = find_centroid(mask, verbose=verbose)
        cz = camera.get_distance(cx, cy)

        if verbose:
            print("Pixels: {} {} {}".format(cx, cy, cz))
        # We convert to camera coordinates
        [x, y, z] = camera.image_2_camera((cx, cy), cz)  # in mm here

        return x, y, z, np.array(plane.vector)

    else :
        # If the mask is empty, then we cannot compute the orientation and the position
        raise ValueError('Mask is empty, cannot compute flower pose.')


def pred_to_pose(p, img, camera, mode='img', radius = 3, conf = 0.25, verbose=False):
    """
    From predictions returns flower poses (coordinates and corolla orientation if possible)
    :param p: predictions for img. dictionary of tensors : 'boxes', 'labels', 'scores' (opt: 'masks')
    :param img: np.ndarray of dimensions (3, H, W)
    :param camera: Camera object
    :param mode: mode of orientation computation. Can be either 'mask' or 'img'
    :param radius: (int) for secondary depth assessment method
    :param conf: (double) prediction confidence threshold
    :return: a dictionary of fields:
            - 'centroids_coo':  flowers center coordinates and score (np.ndarray of dim (N x 4)
            - 'nb_detected' : number of flowers detected (int)
            - 'orientation': (np.ndarray of dim(N,3)) plane vectors of flowers corolla orientation
    """

    ny, nx = img.shape[0], img.shape[1]
    centroids =[] # will be of size Nb correct detection x 4
    masks = [] # to build HSV masks if needed
    orientations = [] # will be of size Nb correct detection x 3

    # Build HSV masks if not already present (present only if model was Mask RCNN)
    if 'masks' in p:
        for mask in p['masks']:
            masks.append(np.asarray(mask.squeeze(0).detach().cpu())*255)
        p['masks'] = masks

    else:
        for b, score in zip(p['boxes'], p['scores']):
            masks.append(get_mask(img, b, score=score, conf=conf))
        p['masks'] = masks

    for box, score, mask in zip(p['boxes'], p['scores'], p['masks']):

        # We skip detections below confidence threshold
        if score < conf:
            continue

        else:

            try:

                # Find object position in camera coordinates (in mm) and orientation
                x, y, z, orientation = get_pose_and_orientation(mask, camera, verbose=verbose)
                orientations.append(orientation)
                if verbose:
                    print("Coordinates: {} mm, {} mm, {} mm".format(x,y,z))
                    print("Plane: {}".format(orientation))

                # in m instead of mm
                centroids.append([x/1000, y/1000, z/1000, score])


            # If HSV thresholding didn't work
            except(ValueError):

                if verbose: print('Error: mask could not be computed, turned to approximate flower location')

                # We compute the center of the bounding box which should correspond to the flower location
                cx = int(((box[2] + box[0]) / 2).numpy())
                cy = int(((box[1] + box[3]) / 2).numpy())

                # We retrieve the depth pixels around this center and we compute the mean depth to approximate flower depth
                depth_frame = camera.depth_frame
                my, My, mx, Mx = max(0, cy - radius), min(ny, cy + radius), max(0, cx - radius), min(nx, cx + radius)
                depth = depth_frame[ my:My, mx:Mx]
                cz = depth.mean()
                # Could not compute the orientation
                orientations.append([0, 0, 0])

                # We convert from pixels to camera coordinates
                [x, y, z] = camera.image_2_camera([cx, cy], cz)
                # in m instead of mm
                centroids.append([x/1000, y/1000, z/1000, score])

    if centroids :
        return {'centroids_coo': np.array(centroids), 'nb_detected': len(centroids), 'orientations': orientations}, True

    else: # If centroids is empty, then no detections
        print("No detections above threshold")
        return None, False