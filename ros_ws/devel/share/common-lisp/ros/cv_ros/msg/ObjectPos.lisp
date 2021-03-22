; Auto-generated. Do not edit!


(cl:in-package cv_ros-msg)


;//! \htmlinclude ObjectPos.msg.html

(cl:defclass <ObjectPos> (roslisp-msg-protocol:ros-message)
  ((centroid
    :reader centroid
    :initarg :centroid
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (plane_vector
    :reader plane_vector
    :initarg :plane_vector
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass ObjectPos (<ObjectPos>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ObjectPos>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ObjectPos)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name cv_ros-msg:<ObjectPos> is deprecated: use cv_ros-msg:ObjectPos instead.")))

(cl:ensure-generic-function 'centroid-val :lambda-list '(m))
(cl:defmethod centroid-val ((m <ObjectPos>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader cv_ros-msg:centroid-val is deprecated.  Use cv_ros-msg:centroid instead.")
  (centroid m))

(cl:ensure-generic-function 'plane_vector-val :lambda-list '(m))
(cl:defmethod plane_vector-val ((m <ObjectPos>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader cv_ros-msg:plane_vector-val is deprecated.  Use cv_ros-msg:plane_vector instead.")
  (plane_vector m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ObjectPos>) ostream)
  "Serializes a message object of type '<ObjectPos>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'centroid))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'plane_vector))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ObjectPos>) istream)
  "Deserializes a message object of type '<ObjectPos>"
  (cl:setf (cl:slot-value msg 'centroid) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'centroid)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'plane_vector) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'plane_vector)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ObjectPos>)))
  "Returns string type for a message object of type '<ObjectPos>"
  "cv_ros/ObjectPos")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ObjectPos)))
  "Returns string type for a message object of type 'ObjectPos"
  "cv_ros/ObjectPos")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ObjectPos>)))
  "Returns md5sum for a message object of type '<ObjectPos>"
  "c5326c21eb2d7acd7c7195e7a2cc19f0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ObjectPos)))
  "Returns md5sum for a message object of type 'ObjectPos"
  "c5326c21eb2d7acd7c7195e7a2cc19f0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ObjectPos>)))
  "Returns full string definition for message of type '<ObjectPos>"
  (cl:format cl:nil "float64[3] centroid ~%float64[3] plane_vector~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ObjectPos)))
  "Returns full string definition for message of type 'ObjectPos"
  (cl:format cl:nil "float64[3] centroid ~%float64[3] plane_vector~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ObjectPos>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'centroid) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'plane_vector) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ObjectPos>))
  "Converts a ROS message object to a list"
  (cl:list 'ObjectPos
    (cl:cons ':centroid (centroid msg))
    (cl:cons ':plane_vector (plane_vector msg))
))
