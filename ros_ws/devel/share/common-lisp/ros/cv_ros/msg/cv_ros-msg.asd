
(cl:in-package :asdf)

(defsystem "cv_ros-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ObjectPos" :depends-on ("_package_ObjectPos"))
    (:file "_package_ObjectPos" :depends-on ("_package"))
  ))