import cv2
import numpy as np
import registration as ro
img0 = ro.readimg( './light/DSC_4364.tiff.fix.tiff')
img1 = ro.readimg( './light/DSC_4384.tiff.fix.tiff')
kp0, des0 = ro.detector(img0)
kp1, des1 = ro.detector(img1)

