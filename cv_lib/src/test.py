# -*- coding: utf-8 -*-
"""
Run test to check if tracker shows correct behavior
"""
import os
import numpy as np
from tracker import Tracker

currentPath = os.path.dirname(os.path.realpath(__file__))


def track_test():
    tracker = Tracker(0.1)

    detection_1 = [
        np.array([[1,2,3],
                  [4,5,6]], dtype=float),

        np.array([[0.1, 0.3,0.6],
                  [0.9, 0.8, 0.7]], dtype=float)
    ]

    detection_2 = [
        np.array([[1.1, 2.2, 3.3],
                  [4, 5, 6]], dtype=float),

        np.array([[0.1, 0.3, 0.6],
                  [0.9, 0.8, 0.7]], dtype=float)

    ]
    # Test 4
    test4_0 = np.array([[1,2,3],
                        [4,5,6],
                        [1,2,3],
                        [4,5,6]])
    print(test4_0.shape)
    test4_1 = np.array([[0.1, 0.3,0.6],
                        [0.9, 0.8, 0.7],
                        [1, 2, 3],
                        [4, 5, 6]])

    # Test 5
    test5_0 = np.vstack((test4_0, np.array([1.1, 2.2, 3.3])))
    test5_1 = np.vstack((test4_1, np.array([0.1, 0.3,0.6])))

    print('detection 1 foo')
    tracker.update(detection_1[0], detection_1[1], 'foo')

    # Test 1
    print('Test if correctly stored')
    obj_list = tracker.get_object_list() # OK
    np.testing.assert_array_equal(obj_list[1], detection_1[0])
    np.testing.assert_array_equal(obj_list[2], detection_1[1])

    # Test2 :
    print('Test if equivalent to call above')
    obj_list_2, type = tracker.get_object_list(type = 'foo')
    np.testing.assert_array_equal(obj_list_2[0], detection_1[0])
    np.testing.assert_array_equal(obj_list_2[1], detection_1[1])

    # Test 3:
    print('Add new and make sure that only new is returned')

    print('detection 1 bar')
    tracker.update(detection_1[0], detection_1[0], 'bar')
    obj_list_3, type = tracker.get_object_list(type='bar')
    np.testing.assert_array_equal(obj_list_3[0], detection_1[0])
    np.testing.assert_array_equal(obj_list_3[1], detection_1[0])

    # Test 4
    print('Make sure every body is returned')
    obj_list_4 = tracker.get_object_list()
    np.testing.assert_array_equal(obj_list_4[1], test4_0)
    np.testing.assert_array_equal(obj_list_4[2], test4_1)

    # Test 5
    print('detection 2 foo')
    tracker.update(detection_2[0], detection_2[1], 'foo')
    obj_list_5 = tracker.get_object_list()
    np.testing.assert_array_equal(obj_list_5[1], test5_0)
    np.testing.assert_array_equal(obj_list_5[2], test5_1)

    # Test 6
    assert(tracker.nb_detected['foo'] == 3)
    assert(tracker.nb_detected['bar'] == 2)

if __name__ == '__main__':
    track_test()
