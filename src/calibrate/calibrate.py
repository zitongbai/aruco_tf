#! /usr/bin/env python3

"""calibrate.py
Calibrate camera using chessboard pattern.
Ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""

import cv2
import numpy as np
import os

img_folder = os.path.join(os.getcwd(), "calib_pic")


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w = 9   # 10 - 1, YOU MIGHT NEED TO CHANGE THIS
h = 6   # 7 - 1, YOU MIGHT NEED TO CHANGE THIS
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 19.00 # 19mm, YOU MIGHT NEED TO CHANGE THIS

objpoints = []
imgpoints = []

images = os.listdir(img_folder)
for img_name in images:
    img = cv2.imread(os.path.join(img_folder, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (w, h), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(50)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret: ", ret)
print("mtx: ", mtx) # intrinsic matrix
print("dist: ", dist)   # distortion coefficients

np.savez("calib.npz", mtx=mtx, dist=dist)