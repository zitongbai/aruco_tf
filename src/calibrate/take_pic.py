#! /usr/bin/env python3

import cv2
import os

# select camera
cam = cv2.VideoCapture(0)

cv2.namedWindow("take_pic")

img_counter = 0
img_folder = os.path.join(os.getcwd(), "calib_pic")
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

while True:
    ret, frame = cam.read()
    cv2.imshow("take_pic", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "calib_{}.png".format(img_counter)
        img_path = os.path.join(img_folder, img_name)
        cv2.imwrite(img_path, frame)
        print("{} written!".format(img_path))
        img_counter += 1