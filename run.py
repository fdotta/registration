import cv2
import numpy as np
import registration as ro
import time
import drawtria as dt

t0 = time.time()
img0 = ro.readimg( './light/DSC_4364.tiff.fix.tiff')
img1 = ro.readimg( './light/DSC_4384.tiff.fix.tiff')
# kp0, des0 = ro.detector(img0)
# kp1, des1 = ro.detector(img1)
kp0, des0 = ro.detector(img0, mfeat='BRISK', mcomp='BRISK')
kp1, des1 = ro.detector(img1, mfeat='BRISK', mcomp='BRISK')
print "time: %5.2f [kp, des]" % (time.time() - t0)

t0 = time.time()
ptf0, vf0, af0, minarea = ro.ftria1(kp0)
print "time: %5.2f [tria1]" % (time.time() - t0)

t0 = time.time()
t00 = t0
ptf1, vf1, af1 = ro.ftria2(kp1, minarea)
print "time: %5.2f [tria2]" % (time.time() - t0)

t0 = time.time()
ptm0, ptm1, nf = ro.ftriam(ptf0, ptf1, vf0, vf1, af0, af1)
print "time: %5.2f [match]" % (time.time() - t0)

t0 = time.time()
ptm0o, ptm1o, TM, T = ro.Tmatrix(ptm0, ptm1)
print "time: %5.2f [TM]" % (time.time() - t0)

t0 = time.time()
timg1 = cv2.warpAffine(img1, TM, (img1.shape[1], img1.shape[0]))
print "time: %5.2f [warp]" % (time.time() - t0)
print "time: %5.2f [per img]" % (time.time() - t00)
cv2.imwrite('warp_img0_0.jpg', img0)
cv2.imwrite('warp_img1_t.jpg', timg1)

ro.drawkeypts(img0, kp0, 'img0')
ro.drawkeypts(img0, kp0, 'img1')

dt.drawtria(img1, ptm1o[:,:,0], "t01_img1")
dt.drawtria(img1, ptm1o[:,:,1], "t02_img1")
dt.drawtria(img0, ptm0o[:,:,0], "t01_img0")
dt.drawtria(img0, ptm0o[:,:,1], "t02_img0")

dt.drawtria(img1, ptm1o[:,:,2], "t03_img1")
dt.drawtria(img1, ptm1o[:,:,3], "t04_img1")
dt.drawtria(img0, ptm0o[:,:,2], "t03_img0")
dt.drawtria(img0, ptm0o[:,:,3], "t04_img0")

dt.drawtria(img1, ptm1o[:,:,4], "t05_img1")
dt.drawtria(img1, ptm1o[:,:,5], "t06_img1")
dt.drawtria(img0, ptm0o[:,:,4], "t05_img0")
dt.drawtria(img0, ptm0o[:,:,5], "t06_img0")

dt.drawtria(img1, ptm1o[:,:,24], "t25_img1")
dt.drawtria(img1, ptm1o[:,:,25], "t26_img1")
dt.drawtria(img0, ptm0o[:,:,24], "t25_img0")
dt.drawtria(img0, ptm0o[:,:,25], "t26_img0")

dt.drawtria(img1, ptm1o[:,:,44], "t45_img1")
dt.drawtria(img1, ptm1o[:,:,45], "t46_img1")
dt.drawtria(img0, ptm0o[:,:,44], "t45_img0")
dt.drawtria(img0, ptm0o[:,:,45], "t46_img0")

dt.drawtria(img1, ptm1o[:,:,64], "t65_img1")
dt.drawtria(img1, ptm1o[:,:,65], "t66_img1")
dt.drawtria(img0, ptm0o[:,:,64], "t65_img0")
dt.drawtria(img0, ptm0o[:,:,65], "t66_img0")


