#! /usr/bin/env python3

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from tf2_ros import TransformBroadcaster

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import math

def quaternion_from_rotation_vector(rvec):
    """
    Convert rotation vector to quaternion
    :param rvec: rotation vector, e.g. [1,2,3]
    :return: quaternion, [x,y,z,w]
    """
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    else:
        axis = rvec / theta
        return np.array([np.sin(theta / 2) * axis[0], np.sin(theta / 2) * axis[1],
                         np.sin(theta / 2) * axis[2], np.cos(theta / 2)], dtype=np.float32)



class DetectAruco:
    def __init__(self) -> None:
        self.camera_info = CameraInfo
        self.image = None
        
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        self.bridge = CvBridge()
        
        # tf broadcaster
        self.tf_broadcaster = TransformBroadcaster()
        
        self.aruco_id = 2
        self.aruco_length = 0.1
        
        
    def camera_info_callback(self, msg):
        self.camera_info = msg
        
    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_draw = self.image.copy()
        
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100) # 5x5 aruco marker, 100 ids
        aruco_params = cv2.aruco.DetectorParameters_create()    # default parameters
        # detect aruco marker
        # corners: list of 4 corners of each marker
        # ids: list of ids of each marker
        corners, ids, rejected = cv2.aruco.detectMarkers(self.image, aruco_dict, parameters=aruco_params) 
        
        if ids is not None and self.camera_info is not None:
            self.camera_info: CameraInfo

            marker_border_color = (0, 255, 0)
            cv2.aruco.drawDetectedMarkers(img_draw, corners, ids, marker_border_color)
            camera_intrinsic = np.array(self.camera_info.K, dtype=np.float32).reshape((3, 3))
            distortion_parameter = np.array(self.camera_info.D, dtype=np.float32).reshape((1, 5))
            
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.aruco_length, camera_intrinsic, distortion_parameter)
            cv2.aruco.drawAxis(img_draw, camera_intrinsic, distortion_parameter, rvec, tvec, 0.03)
            
            # only publish tf of the specified aruco marker
            if self.aruco_id in ids:
                idx = np.where(ids == self.aruco_id)[0][0]
                rvec = rvec[idx] # rotation vector, shape: (1, 3), e.g. [[-2.76186719  0.23810233 -0.76484692]]
                tvec = tvec[idx] # translation vector, shape: (1, 3), e.g. [[-0.0084606  -0.01231327  0.26638986]]
                rvec = np.squeeze(rvec) # shape: (3, ), e.g. [-2.76186719  0.23810233 -0.76484692]
                tvec = np.squeeze(tvec) # shape: (3, ), e.g. [-0.0084606  -0.01231327  0.26638986]
                # convert rotation vector to quaternion
                quat = quaternion_from_rotation_vector(rvec)
                # prepare tf message
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "camera_color_frame"
                t.child_frame_id = "aruco_marker"
                
                t.transform.translation.x = tvec[0]
                t.transform.translation.y = tvec[1]
                t.transform.translation.z = tvec[2]
                t.transform.rotation.x = quat[0]
                t.transform.rotation.y = quat[1]
                t.transform.rotation.z = quat[2]
                t.transform.rotation.w = quat[3]
                
                # send tf
                self.tf_broadcaster.sendTransform(t)
                
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', img_draw.shape[1], img_draw.shape[0])
        cv2.imshow('image', img_draw)
        cv2.waitKey(1)
        
def main():
    rospy.init_node("detect_aruco", anonymous=True)
    detect_aruco = DetectAruco()
    rospy.spin()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()