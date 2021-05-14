from myUtils import*
import time
from skspatial.objects import Points, Plane


class ObjectPredictor():
    def __init__(self, model_name = 'MobileNetV3_largeFPN', weights_path = '/home/phil/Documents/Projects/CV_SLAM_project/cv_lib/models/', classes=2):

        assert model_name in ['Resnet50_FPN',
                              'MobileNetV3_largeFPN',
                              'MobileNetV3_largeFPN_320',
                              'MaskRCNN',
                              'YOLOv5x'],\
            "Error: model not found"

        # Load model
        weights_path += 'best_' + model_name + ".pt"
        self.model = get_object_detection_model(num_classes=classes, mtype= model_name, weights_path= weights_path, \
                                                device=torch.device('cpu'))
        self.model_name = model_name
        # Transform
        self.transform = get_img_transformed(train=False)


    def __call__(self, image, camera, conf=0.5, verbose=False):
        assert isinstance(image, np.ndarray), "Error: img should be of type np.ndarray"

        # Get image and convert it to handled type
        if self.model_name != 'YOLOv5x':
            img = self.transform(image).unsqueeze(0).to(torch.device('cpu')) # convert to tensor of size [1, 3, H, w]
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # TODO cv2.resize(image, (640,640))

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
            print(pred[0])
            print("{} flowers detected.".format(len(pred[0]['scores'] > conf)))
            draw_bboxes(pred[0], image, conf=conf)

        poses = pred_to_pose(pred[0], image, camera, conf=conf, verbose=verbose)

        if verbose: print("Elapsed time overall {} s".format(time.time() - t1))


        return poses


########################################################################################################################
# Functions
########################################################################################################################

def get_mask(img, score, it=2, conf=0.3, verbose=False):
    """

    :param img: image from which we want to detect a white flower, ndarray of size (3, H, W)
    :param it: number of iterations for mask erosion
    :return: get mask of white elements in the image
    """
    output = img.copy()
    output = cv2.medianBlur(output, 5)

    # Define HBR white color range for thresholding TUNED :)
    low_orange = np.array([0,0,228], dtype=np.float32)
    upp_orange = np.array([180,15,255], dtype=np.float32)

    # Opening to get rid of the small background artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # From bgr to hsv colorspace
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    # Threshold and erode
    mask = cv2.inRange(hsv, low_orange, upp_orange)
    mask = cv2.erode(mask, kernel, iterations=it)

    if verbose and score >= conf :
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask/255

def crop(img, bbox):
    """

    :param img: ndarray of size (3, H, W)
    :param bbox: torch tensor [xm, ym, xM, yM]
    :return: cropped picture
    """
    [xm, ym, xM, yM] = bbox.numpy().astype(int)

    return img[ym:yM, xm:xM,:]

def get_orientation(pred, img, camera, mode='mask', conf=0.5, verbose=False):

    """

    :param pred: dictionnary of torch tensors : 'boxes', 'labels', 'scores' ('masks' optional)
    :param img: ndarray of size (3, H, W). contains flowers.
    :param camera: camera object
    :param mode: 'mask': if we retrieve orientation from already computed mask, 'img': if we need to compute the mask
    :param conf: confidence of predictions
    :return: flowers corolla orientations (ndarray of size (Nb_detection, 3))
    """

    orientations = []
    if mode == 'mask':
        orientations = get_plane_orientation(pred, camera, conf=conf)
    if mode == 'img':
        masks = []
        for b, score in zip(pred['boxes'], pred['scores']):
            masks.append(get_mask(crop(img,b), verbose=verbose, score=score, conf=conf))

        pred['masks'] = masks
        orientations = get_plane_orientation(pred, camera, conf=conf)

    return np.array(orientations)


def pred_to_pose(p, img, camera, mode='img', radius = 3, conf = 0.25, verbose=False):
    """
    From predictions returns flower poses (coordinates and corolla orientation if possible)
    :param p: prediction dictionary of tensors : 'boxes', 'labels', 'scores' (opt: 'masks')
    :param img: np.ndarray of dimensions (H,W,3)
    :param camera: Camera object
    :param mode: mode of depth retrieval and orientation computation. Can be either 'mask' or 'img'
    :param radius: useful for 'img' mode depth estimation
    :param conf: (double) prediction confidence threshold
    :return: a dictionary of fields:
            - 'centroids_coo':  flowers center coordinates and score (np.ndarray of dim (N x 4)
            - 'nb_detected' : number of flowers detected (int)
            - 'orientation': (np.ndarray of dim(N,3)) plane vectors of flowers corolla orientation
    """

    ny, nx = img.shape[0], img.shape[1]
    centroids =[] # will be of size Nb correct detection x 4
    if 'masks' not in p:
        p['masks'] = torch.zeros((len(p['labels']),ny,nx))

    for box, score, mask in zip(p['boxes'], p['scores'], p['masks']):

        cx = int(((box[2] + box[0])/2).numpy())
        cy = int(((box[1] + box[3])/2).numpy())
        depth_frame = camera.depth_frame
        # remember: img has shape (H,W,3) where H:ny, W:nx
        my, My, mx, Mx = max(0,cy-radius), min(ny, cy+radius), max(0, cx-radius), min(nx, cx+radius)
        depth = depth_frame[ my:My, mx:Mx]
        cz = depth.mean()

        [x, y, z] = camera.image_2_camera([cx / 1000, cy / 1000], cz / 1000) # in meters instead of mm
        if score >= conf:
            print("Computed centroids")
            centroids.append([x, y, z, score])

    orientations = get_orientation(p, img, camera, mode=mode, conf=conf, verbose=verbose) # of size Nb correct detection x 3
    #orientations = np.zeros((len(centroids),3))

    if centroids :
        return {'centroids_coo': np.array(centroids), 'nb_detected': len(p['scores'] >= conf), 'orientations': orientations}, True
    else:
        print("No detections above threshold")
        return None, False

def get_plane_orientation(pred, camera, conf=0.5):
        """
        Computes normal of object plane and plot
        Args: RS_camera object
        Return: list of plane vectors if predictions are above confidence conf
        """
        # TODO: handle error case (mask is empty)
        # Set plot
        thresh = 1e-17 # ~0

        # For each region, compute plane

        orientations = []
        for mask, score in zip(pred['masks'], pred['scores']):

            if score >= conf :
                # Compute 3D coordinates of all pixels within the object
                obj_pix = np.argwhere(mask*255 > thresh)  # y,x
                if obj_pix.size < 1:
                    orientations.append(np.array[None, None, None])
                    continue
                points = np.empty((obj_pix.shape[0], 3))
                points[:, 0] = obj_pix[:, 1]
                points[:, 1] = obj_pix[:, 0]
                points[:, 2] = camera.depth_frame[obj_pix[:, 0], obj_pix[:, 1]]

                # Feed into Points, best fit etc...
                pts = Points(points)  # must be built with a nd.array
                plane = Plane.best_fit(pts)
                # Append computed plane
                orientations.append(np.array(plane.vector))

        return orientations
