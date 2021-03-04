import cv2

# relevant for trials
def reader(name): #OK
    """Loads image from its name
    Args : name of img which must be located in '../data/'"""
    img = cv2.imread("../data/"+name)
    return img