# Object-pose-estimation
This project is a ROS package `cv_ros` and provides a framework for object detection and pose estimation via HSV thresholding and neural network inference, within the scope of the GrowBotHub-2021 project. 

## Table of contents
* [Requirements](#requirements)
* [Folder hierarchy](#folder-hierarchy)
* [Usage](#usage)
* [More references](#more-references)

## Requirements

### General
- open-cv
- rospy
- scikit-spatial
- pyrealsense2
- realsense2
- cv_bridge
- (matplotlib)
- ros noetic distribution
- python 3.8

### Requirements YOLO network
References : https://github.com/ultralytics/yolov5

- matplotlib >= 3.2.2
- numpy >= 1.18.5 
- opencv-python >= 4.1.2
- PyYAML >= 5.3.1
- scipy >= 1.4.1
- torch >= 1.7.0
- torchvision >= 0.8.1
- tqdm >= 4.41.0

### Hardware

Intel® RealSense™ D415 stereo camera : https://www.intelrealsense.com/depth-camera-d415/

## Folder hierarchy

```
Object-pose-estimation/
├── __init__.py
├── README.md
├── Report.pdf
├── cv_lib
│   ├── __init__.py
│   ├── models
│   │   └── best_MobileNetV3_largeFPN.pt
│   └── src
│       ├── camera_listener.py
│       ├── myClasses.py
│       ├── myTransforms.py
│       ├── myUtils.py
│       ├── object_detection.py
│       ├── object_prediction.py
│       ├── rs_camera.py
│       ├── take_pictures.py
│       ├── test.py
│       └── tracker.py
└── ros_ws
    ├── __init__.py
    └── src
        ├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
        ├── cv_ros
        │   ├── CMakeLists.txt
        │   ├── __init__.py
        │   ├── msg
        │   │   ├── __init__.py
        │   │   └── ObjectPose.msg
        │   ├── package.xml
        │   └── src
        │       ├── __init__.py
        │       ├── main_ros.py
        │       ├── test_main.py
        │       └── tuning.py
        └── __init__.py

```

## Usage

### What this package provides
* Colorful object detection and pose estimation via HSV thresholding and depth data
* Strawberry flower detection and pose estimation via neural network inference, HSV thresholding and depth data

Each step of object detection can be displayed (set verbose to True), and parameters can be tuned within `main_ros.py`. More details in the string documentation at the beginning of the file.

### Launch node
To launch our package, one needs to run the following commands in several terminals, starting from ```Object-pose-estimation``` as home directory.
Only once to compile :
```
$ cd ros_ws
$ catkin_make
```
All terminals :
```
$ cd ros_ws
$ source devel/setup.bash
```
Terminal 1: Start ROS Master
```
$ roscore
```
Terminal 2: Publish start message for our \Vision node
```
$ rostopic pub /RS_status std_msgs/String "STANDSTILL" # To start the node
```
Terminal 3: Launch camera
```
$ roslaunch realsense2_camera rs_camera.launch aligned_depth:=true
```
Terminal 4: Launch our ROS node from our package
```
$ rosrun cv_ros main_ros.py
```

### ROS topics to listen to
\Vision node will publish on the following topics:
- \CV_data : ObjectPose messages. Here you can retrieve the detected object list and their pose.
- \CV_status : RETRY or SUCCESS. Indicates whether detection has been successful or not
- \CV_object : type and number of detections

### HSV tuning
This will open a window to find correct HSV lower and upper threshold for optimized object detection.

All terminals :
```
$ cd ros_ws
$ source devel/setup.bash
```
Terminal 1: Start ROS Master
```
$ roscore
```
Terminal 2: Launch camera
```
$ roslaunch realsense2_camera rs_camera.launch aligned_depth:=true
```
Terminal 3: Launch our ROS tuning node from our package
```
$ rosrun cv_ros tuning.py
```


## More references
* Camera ROS tutorials : https://github.com/IntelRealSense/realsense-ros#installation-instructions, http://wiki.ros.org/realsense2_camera
* ROS tutorials : http://wiki.ros.org/
* Robotics-2021 project integration https://github.com/GrowbotHub/Robotics-2021.
  


