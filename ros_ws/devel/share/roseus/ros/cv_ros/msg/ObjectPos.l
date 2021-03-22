;; Auto-generated. Do not edit!


(when (boundp 'cv_ros::ObjectPos)
  (if (not (find-package "CV_ROS"))
    (make-package "CV_ROS"))
  (shadow 'ObjectPos (find-package "CV_ROS")))
(unless (find-package "CV_ROS::OBJECTPOS")
  (make-package "CV_ROS::OBJECTPOS"))

(in-package "ROS")
;;//! \htmlinclude ObjectPos.msg.html


(defclass cv_ros::ObjectPos
  :super ros::object
  :slots (_centroid _plane_vector ))

(defmethod cv_ros::ObjectPos
  (:init
   (&key
    ((:centroid __centroid) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:plane_vector __plane_vector) (make-array 3 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _centroid __centroid)
   (setq _plane_vector __plane_vector)
   self)
  (:centroid
   (&optional __centroid)
   (if __centroid (setq _centroid __centroid)) _centroid)
  (:plane_vector
   (&optional __plane_vector)
   (if __plane_vector (setq _plane_vector __plane_vector)) _plane_vector)
  (:serialization-length
   ()
   (+
    ;; float64[3] _centroid
    (* 8    3)
    ;; float64[3] _plane_vector
    (* 8    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[3] _centroid
     (dotimes (i 3)
       (sys::poke (elt _centroid i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _plane_vector
     (dotimes (i 3)
       (sys::poke (elt _plane_vector i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[3] _centroid
   (dotimes (i (length _centroid))
     (setf (elt _centroid i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _plane_vector
   (dotimes (i (length _plane_vector))
     (setf (elt _plane_vector i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get cv_ros::ObjectPos :md5sum-) "c5326c21eb2d7acd7c7195e7a2cc19f0")
(setf (get cv_ros::ObjectPos :datatype-) "cv_ros/ObjectPos")
(setf (get cv_ros::ObjectPos :definition-)
      "float64[3] centroid 
float64[3] plane_vector

")



(provide :cv_ros/ObjectPos "c5326c21eb2d7acd7c7195e7a2cc19f0")


