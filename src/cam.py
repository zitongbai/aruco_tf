#! /usr/bin/env python3

import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo

class CameraPC:
    """Camera class for personal computer
    Get image from PC camera and publish it and camera info
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        self.bridge = CvBridge()
        
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=1)
        
        # self.intrinsic_matrix = [975.95379655, 0, 628.97795209, 0, 978.75260475, 359.97734358, 0, 0, 1]
        # self.distortion_coefficients = [0.1381094, -0.40191369, 0.0026172, -0.0038425, 0.24211502]
        
        # get the intrinsic matrix and distortion coefficients from calib.npz
        current_folder = os.path.dirname(os.path.abspath(__file__))
        calib_file = os.path.join(current_folder, "calibrate", "calib.npz")
        rospy.loginfo("loadinf calibration file: %s", calib_file)
        # get intrinsic matrix and distortion coefficients
        if os.path.exists(calib_file):
            calib = np.load(calib_file)
            self.intrinsic_matrix = calib["mtx"]
            self.distortion_coefficients = calib["dist"]
            # flatten into 1D array
            self.intrinsic_matrix = self.intrinsic_matrix.flatten()
            self.distortion_coefficients = self.distortion_coefficients.flatten()
        else:
            print("calib.npz not found")
        
    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(img_msg)
            
            camera_info = CameraInfo()
            camera_info.header.stamp = rospy.Time.now()
            camera_info.width = frame.shape[1]
            camera_info.height = frame.shape[0]
            camera_info.distortion_model = "plumb_bob"
            camera_info.K = self.intrinsic_matrix
            camera_info.D = self.distortion_coefficients
            self.camera_info_pub.publish(camera_info)
            
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            rate.sleep()
        
        self.cap.release()
        
if __name__ == "__main__":
    rospy.init_node("camera_pc", anonymous=True)
    
    camera_pc = CameraPC()
    camera_pc.run()
    
    