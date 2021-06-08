import pandas as pd
import cv2
import numpy as np
import torchvision.transforms as T
import torchvision, torch
import os
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.functional as F
#from google.colab.patches import cv2_imshow

################################################################################################################
# SOME CONSTANTS
################################################################################################################

MEAN_Imagenet = [0.485, 0.456, 0.406]
STD_Imagenet = [0.229, 0.224, 0.225]

################################################################################################################
# VISUALIZATIONS
################################################################################################################

def draw_bboxes(target, image, conf = 0.5, scale = 1, name=''):
    """
    Draws bounding boxes around predictions
    :param target: prediction dictionnary with fileds : 'boxes', 'labels', 'scores
    :param image: ndarray of dimensions [H, W, 3]
    :param conf: confidence score threshold below which detection is not considered
    :param name: to save the detections picture
    """

    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    full_name = "predictions/" + name + "_predictions.jpg"

    for [xm,ym,xM,yM], label, score in zip(target["boxes"], target["labels"], target["scores"]):
      if label == 1: c = (255,0,0)
      else: c = (0,255,0)
      if score > conf :
        img = cv2.rectangle(img, (int(xm),int(ym)), (int(xM),int(yM)), c, 4)
        cv2.putText(img, str(np.trunc(score.item()*100)), (int(xm), int(yM)), font, 0.5, c, 1, cv2.LINE_AA)
        print("flower score: ", score.item())
    
    rescale = (int(image.shape[1]/scale), int(image.shape[0]/scale))
    img = cv2.resize(img, rescale)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(full_name, img)

def draw_corners(image, corners):
  img = image.copy()
  for [a,b,c,d,e,f,g,h] in corners:
      img = cv2.circle(img, (a,b), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (c,d), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (e,f), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (g,h), radius=1, color=(255, 255, 255), thickness=2)
  #cv2_imshow(img)
  cv2.imshow("Image rotated with corners",img)

def nothing(x):
    pass

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

        cv2.imshow('Mask_tuning',mask)
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
    cv2.destroyAllWindows()

def create_canny_slidebar(img, scale = 1):

    """
    Creates a trackbar to find threshold fo canny edge detection
    It was just a trial slidebar
    """

    def kmeans(img, nb=3):
        output = img.copy()
        clust = output.reshape((-1, 3))
        clust = np.float32(clust)  # should be flattened and of type float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # max iter and accuracy epsilon
        K = nb
        ret, label, center = cv2.kmeans(clust, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        clust = res.reshape((output.shape))

        return clust

    def get_contour(img, thresh=100):
        contours, _ = cv2.findContours(img,)

    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    output = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    edge = np.zeros(output.shape, np.uint8)
    else_img = np.zeros_like(output)

    cv2.namedWindow('Canny threshold')

    # create trackbars for color change
    cv2.createTrackbar('L', 'Canny threshold', 0, 255, nothing)
    cv2.createTrackbar('H', 'Canny threshold', 0, 255, nothing)
    cv2.createTrackbar('k', 'Canny threshold', 2,30, nothing)
    cv2.createTrackbar('t', 'Canny threshold', 2,10, nothing)

    # create switch for ON/OFF functionality
    blur = '0 : OFF \n1 : ON'
    cv2.createTrackbar(blur, 'Canny threshold', 0, 1, nothing)

    while (1):
        img_ = np.vstack((edge, else_img))
        cv2.imshow('Canny threshold', img_)

        wk = cv2.waitKey(1) & 0xFF
        if wk == 27:
            break

        # get current positions of four trackbars
        l = cv2.getTrackbarPos('L', 'Canny threshold')
        h = cv2.getTrackbarPos('H', 'Canny threshold')
        b = cv2.getTrackbarPos(blur, 'Canny threshold')
        t = cv2.getTrackbarPos('t', 'Canny threshold')
        k = cv2.getTrackbarPos('k', 'Canny threshold')

        if b == 1 :
            else_img = cv2.blur(output, (k, k))
            edge = cv2.Canny(else_img, l, h)
        else:
            else_img = kmeans(output, nb=t)
            edge = cv2.Canny(else_img, l, h)

    cv2.destroyAllWindows()


################################################################################################################
# TRAINING AND TEST UTILS
################################################################################################################

def get_img_transformed(train=False):
  """
  Apply mandatory transforms on the image

  Returns:
            - Composition of transforms
  """
  transforms = []
  # converts the image into a PyTorch Tensor
  transforms.append(T.ToTensor())
  # image scaling and normalization
  transforms.append(T.Normalize(mean=MEAN_Imagenet, std=STD_Imagenet))
  if train:
      transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1))
  return T.Compose(transforms)

def collate_fn(batch):
  """Credits to https://github.com/pytorch/vision/blob/master/references/detection/utils.py"""
  return tuple(zip(*batch))


def bbox_area(bbox):
    """
    Compute bounding boxes area
    :param bboxes: (numpy array of dimensions (nb_boxes, 4)
    :return area: (numpy array of dimensions (nb_boxes,)

    """
    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    return area


def get_object_detection_model(num_classes, device, mtype = 'Resnet50_FPN', weights_path = '../models/best_Resnet50_FPN.pt'):
    # load a model pre-trained on COCO

    if mtype ==  'Resnet50_FPN':
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      model.load_state_dict(torch.load(weights_path, map_location=device))

    if mtype == 'MobileNetV3_largeFPN':

      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      model.load_state_dict(torch.load(weights_path, map_location=device))
    
    if mtype == 'MobileNetV3_largeFPN_320':
      # Low resolution network
      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      model.load_state_dict(torch.load(weights_path, map_location=device))

    if mtype == 'MaskRCNN':
      # load an instance segmentation model pre-trained on COCO
      model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      # now get the number of input features for the mask classifier
      in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
      hidden_layer = 256
      # and replace the mask predictor with a new one
      model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                          hidden_layer,
                                                          num_classes)
      model.load_state_dict(torch.load(weights_path, map_location=device))

    if mtype == 'YOLOv5x':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path =weights_path )
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    return model






