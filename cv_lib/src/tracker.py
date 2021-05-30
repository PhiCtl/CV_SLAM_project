import numpy as np


class Tracker():

    def __init__(self, thres=0.01):

        # To store objects
        self.objects = [np.array([]), # type
                        np.array([], dtype=float).reshape(0,3), # coordinates
                        np.array([], dtype=float).reshape(0,3), # planes
                        np.array([]) # number of times we detected an object
                        ] # list

        # To store number of detection
        self.nb_detected = {} # dict

        self.thresh = thres # defines if two objects are the same or not -> define accuracy

    def update_list(self, coordinate, plane, i):
        """
        If object already detected, updates the list
        :param coordinate: ndarray of dim 3
        :param plane: ndarray of dim 3
        :param i: index of the detected object in the object list objects
        :return:
        """
        self.objects[1][i,:] += coordinate
        self.objects[2][i,:] += plane
        self.objects[3][i] += 1 # udpate the number of times we detected a given object

    def add_object(self, coordinate, plane, type):
        """
        Adds newly detected object to objects list
        :param coordinate: npdarray of dim (3,)
        :param plane: ndarray of dim (3,)
        :param type: (str) object type

        """
        # update number of overall detections
        if not type in self.nb_detected:
            self.nb_detected[type] = 1
        else:
            self.nb_detected[type] += 1

        # append to objects list
        self.objects[0] = np.hstack((self.objects[0], type))
        self.objects[1] = np.vstack((self.objects[1], coordinate))
        self.objects[2] = np.vstack((self.objects[2], plane))
        self.objects[3] = np.hstack((self.objects[3], 1))


    def update(self, coordinates, planes, type):

        """
        Updates tracker's objects list
        :param coordinates: list (of size N: nb of detected elements) of ndarray of dimension 3
        :param planes: list (of size N: nb of detected elements) of ndarray of dimension 3
        :param type: (str) object type

        """

        # For each newly detected object
        for coo, plane in zip(coordinates, planes):

            # find if already in there
            i, found = self.already_in(coo, type)

            # If already in the list
            if found:
                self.update_list(coo, plane, i)
            # If not found
            else:
                self.add_object(coo, plane, type)



    def get_object_list(self, type='any'):

        self.average() # compute the average of the positions and orientations

        if type != 'any':
            # TODO : refine error handling
            # If object type not detected yet
            assert(type in self.objects[0]), \
                "No {} detected so far. \n Available objects are: {}".format(type, set(self.objects[0]))

            # Find objects of interest
            ix = np.argwhere(self.objects[0] == type).flatten()
            return [self.objects[1][ix,:], self.objects[2][ix,:]], type

        else:
            return [self.objects[1], self.objects[2]]

    def already_in(self, coordinate, type):
        """
        Finds closest object in self.objects[1].
        If distance is above self.threshold, then object hasn't been detected until now
        Otherwise, we already detected that object and we return its index in self.objects[1]
        :param coordinate: coordinates of the newly detected object (ndarray of dim 3)
        :return: i : (int) index of the already detected object in self.ojects[1]
                 found: (bool) if object is already in the list
        """

        # If no detections until now, return False
        if self.objects[1].size <= 0 or type not in self.objects[0]:
            return -1, False

        # Otherwise find closest (L2-norm) already detected object
        else:
            ix = np.argwhere(self.objects[0] == type).flatten()
            L2_dist = np.sqrt((self.objects[1][ix,:] - coordinate)**2).sum(axis=1)
            min, i = np.min(L2_dist), np.argmin(L2_dist)

            # If minimum above threshold -> new object has been detected
            if min > self.thresh:
                return -1, False
            # Otherwise, has already been detected
            else:
                return ix[i], True

    def average(self):

        # Does the average type per type
        for type in set(self.objects[0]):

            ix = np.argwhere(self.objects[0] == type).flatten()
            self.objects[1][ix, :] /= np.expand_dims(self.objects[3][ix], axis=1)
            self.objects[2][ix, :] /= np.expand_dims(self.objects[3][ix], axis=1)

        # Reset object detections
        self.objects[3][:] = 1





    def reset(self):
        """
        Resets tracker
        """
        self.objects = [np.array([]),  # type
                        np.array([], dtype=float).reshape(0, 3),  # coordinates
                        np.array([], dtype=float).reshape(0, 3),  # planes
                        np.array([])  # number of times we detected an object
                        ]  # list
        self.nb_detected = {}
