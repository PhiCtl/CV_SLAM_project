from myUtils import*
import time
from skspatial.objects import Points, Plane


class ObjectPredictor():
    def __init__(self, device, model_name = 'MobileNetV3_largeFPN', weights_path = '../models/', classes=2):

        assert model_name in ['Resnet50_FPN', 'MobileNetV3_largeFPN', 'MobileNetV3_largeFPN_320', 'MaskRCNN', 'YOLOv5x'], "Error: model not found"

        # Load model
        self.model = get_object_detection_model(num_classes=classes, mtype= model_name)
        self.device = device
        weights_path += 'best_' + model_name + ".pt"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.mode_name = model_name
        # Transform
        self.transform = get_img_transformed(train=False)

    def __call__(self, image, camera, conf=0.5, verbose=False):
        assert isinstance(image, np.ndarray), "Error: img should be of type np.ndarray"

        # Get image and convert it to handled type
        if self.model_name != 'YOLOv5x':
            img = self.transform(image).unsqueeze(0).to(self.device) # convert to tensor of size [1, 3, H, w]
        else:
            img = image.copy()

        # Make prediction
        self.model.eval()
        t1 = time.time()
        with torch.no_grad():
            pred = self.model(img)
        if verbose:
            print("Elapsed time {} s".format(time.time() - t1))
            print("{} flowers detected.".format(len(pred[0]['scores'] > conf)))

        poses = pred_to_pose(pred[0], image, camera, conf=conf)

        return poses


########################################################################################################################
# Functions
########################################################################################################################

def get_masks(img):
    # TODO implement HSV range
    pass

def get_orientation(pred, img, camera, mode='mask', conf=0.5):

    orientations = []
    if mode == 'mask':
        orientations = get_plane_orientation(pred, camera, conf=conf)
    if mode == 'img':
        pass
    if mode == 'HSV':
        pred['masks'] = get_masks(img)
        orientations = get_plane_orientation(pred, camera, conf=conf)

    return np.array(orientations)


def pred_to_pose(p, img, camera, mode='img', radius = 3, conf = 0.5):
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
        cx = (box[2] + box[0])/2
        cy = (box[1] + box[3])/2
        depth_frame = camera.depth_frame
        # remember: img has shape (H,W,3) where H:ny, W:nx
        depth = depth_frame[ max(0,cy-radius) : min(ny, cy+radius), max(0, cx-radius) : min(nx, cx+radius)]
        cz = depth.mean()

        [x, y, z] = camera.image_2_camera([cx / 1000, cy / 1000], cz / 1000) # in meters instead of mm
        if score >= conf:
            centroids.append([x, y, z, score])

    orientations = get_orientation(p, img, camera, mode=mode, conf=conf) # of size Nb correct detection x 3

    if centroids :
        return {'centroids_coo': np.array(centroids), 'nb_detected': len(p['scores'] >= conf), 'orientations': orientations}, True
    else:
        print("No detections above threshold")
        return None, False

def get_plane_orientation(pred, camera, conf=0.5):
        """
        Computes normal of object plane and plot
        Args: RS_camera object
        Return: list of plane vectors
        """

        # Set plot
        thresh = 1e-17 # ~0

        # For each region, compute plane

        orientations = []
        for mask, score in zip(pred['masks'], pred['scores']):

            if score >= conf :
                # Compute 3D coordinates of all pixels within the object
                obj_pix = np.argwhere(mask*255 > thresh)  # y,x
                points = np.empty((obj_pix.shape[0], 3))
                points[:, 0] = obj_pix[:, 1]
                points[:, 1] = obj_pix[:, 0]
                print(camera.depth_frame.shape, obj_pix.shape, camera.bgr_image.shape)
                points[:, 2] = camera.depth_frame[obj_pix[:, 0], obj_pix[:, 1]]

                # Feed into Points, best fit etc...
                pts = Points(points)  # must be built with a nd.array
                plane = Plane.best_fit(pts)
                # Append computed plane
                orientations.append(np.array(plane.vector))

        return orientations
