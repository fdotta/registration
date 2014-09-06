#/usr/bin/python
import cv2
import numpy as np
import registration as ro
import time
import drawtria as dt
from multiprocessing import Pool
import os,sys


def regrun(args):
    t0 = time.time()
    tiff    = args[0][0]
    des0    = args[1]
    ptf0    = args[2]
    vf0     = args[3]
    af0     = args[4]
    minarea = args[5]
    img1 = ro.readimg(tiff)
    kp1, des1 = ro.detector(img1)
    ptf1, vf1, af1 = ro.ftria2(kp1, minarea)
    lerr = 0.0008
    for i in np.arange(4):
        ptm0, ptm1, nf = ro.ftriam(ptf0, ptf1, vf0, vf1, af0, af1, err=lerr)
        ptm0o, ptm1o, TM, valid = ro.Tmatrix(ptm0, ptm1)
        if not valid:
            lerr = lerr/1.5
        else:
            break
    if valid:
        # fname = './warp/warp_' + os.path.split(tiff)[1]
        fname = './warp/warp_' + os.path.split(os.path.splitext(tiff)[0])[1] + '.jpg'
        timg1 = cv2.warpAffine(img1, TM, (img1.shape[1], img1.shape[0]))
        cv2.imwrite(fname, timg1)
    return time.time() - t0

def findinfolder (path):
    ltiff = []
    for root, dirs, files in os.walk('./light/'):
        for name in files:
            ext = os.path.splitext(name)[1]
            if ext == '.tiff':
                ltiff.append(root + name)
    ltiff.sort()
    return ltiff

def refimg(fname):
    img0 = ro.readimg(fname)
    kp0, des0 = ro.detector(img0)
    ptf0, vf0, af0, minarea = ro.ftria1(kp0)
    # fname = './warp/warp_' + os.path.split(fname)[1]
    fname = './warp/warp_' + os.path.split(os.path.splitext(fname)[0])[1] + '.jpg'
    cv2.imwrite(fname, img0)
    return kp0, des0, ptf0, vf0, af0, minarea

if __name__ == '__main__':
    t0 = time.time()
    ltiffname = findinfolder ('./light')
    print '>>> Ref Data...'
    kp0, des0, ptf0, vf0, af0, minarea = refimg(ltiffname[0])
    print "    time: %5.2f [ref img]" % (time.time() - t0)
    del ltiffname[0]
    print ">>> Registring..."
    nl = len(ltiffname)
    args = []
    for i in np.arange(nl):
        args.append(([ltiffname[i]], des0, ptf0, vf0, af0, minarea))

    # regrun(args[0])
    # print "ok"
    pool = Pool(processes=4)
    ng = 0
    for process in pool.map(regrun, args):
        ng += 1
        print "    time: %5.2f [img%2i]" % (process, ng)
    t1 = time.time()
    print ">>> Total Time %5.2f" % (t1 - t0)
    print ">>> Per Img Time %5.2f" % ((time.time() - t0)/(len(ltiffname)+1))



# t0 = time.time()
# img0 = ro.readimg( './light/DSC_4364.tiff.fix.tiff')
# img1 = ro.readimg( './light/DSC_4384.tiff.fix.tiff')
# kp0, des0 = ro.detector(img0)
# kp1, des1 = ro.detector(img1)
# print "time: %5.2f [kp, des]" % (time.time() - t0)

# t0 = time.time()
# ptf0, vf0, af0, minarea = ro.ftria1(kp0)
# print "time: %5.2f [tria1]" % (time.time() - t0)

# t0 = time.time()
# t00 = t0
# ptf1, vf1, af1 = ro.ftria2(kp1, minarea)
# print "time: %5.2f [tria2]" % (time.time() - t0)

# t0 = time.time()
# ptm0, ptm1, nf = ro.ftriam(ptf0, ptf1, vf0, vf1, af0, af1)
# print "time: %5.2f [match]" % (time.time() - t0)

# t0 = time.time()
# ptm0o, ptm1o, TM, T = ro.Tmatrix(ptm0, ptm1)
# print "time: %5.2f [TM]" % (time.time() - t0)

# t0 = time.time()
# timg1 = cv2.warpAffine(img1, TM, (img1.shape[1], img1.shape[0]))
# print "time: %5.2f [warp]" % (time.time() - t0)
# print "time: %5.2f [per img]" % (time.time() - t00)
# cv2.imwrite('warp_img0_0.jpg', img0)
# cv2.imwrite('warp_img1_t.jpg', timg1)

# ro.drawkeypts(img0, kp0, 'img0')
# ro.drawkeypts(img0, kp0, 'img1')

# dt.drawtria(img1, ptm1o[:,:,0], "t01_img1")
# dt.drawtria(img1, ptm1o[:,:,1], "t02_img1")
# dt.drawtria(img0, ptm0o[:,:,0], "t01_img0")
# dt.drawtria(img0, ptm0o[:,:,1], "t02_img0")

# dt.drawtria(img1, ptm1o[:,:,2], "t03_img1")
# dt.drawtria(img1, ptm1o[:,:,3], "t04_img1")
# dt.drawtria(img0, ptm0o[:,:,2], "t03_img0")
# dt.drawtria(img0, ptm0o[:,:,3], "t04_img0")

# dt.drawtria(img1, ptm1o[:,:,4], "t05_img1")
# dt.drawtria(img1, ptm1o[:,:,5], "t06_img1")
# dt.drawtria(img0, ptm0o[:,:,4], "t05_img0")
# dt.drawtria(img0, ptm0o[:,:,5], "t06_img0")

# dt.drawtria(img1, ptm1o[:,:,24], "t25_img1")
# dt.drawtria(img1, ptm1o[:,:,25], "t26_img1")
# dt.drawtria(img0, ptm0o[:,:,24], "t25_img0")
# dt.drawtria(img0, ptm0o[:,:,25], "t26_img0")

# dt.drawtria(img1, ptm1o[:,:,44], "t45_img1")
# dt.drawtria(img1, ptm1o[:,:,45], "t46_img1")
# dt.drawtria(img0, ptm0o[:,:,44], "t45_img0")
# dt.drawtria(img0, ptm0o[:,:,45], "t46_img0")

# dt.drawtria(img1, ptm1o[:,:,64], "t65_img1")
# dt.drawtria(img1, ptm1o[:,:,65], "t66_img1")
# dt.drawtria(img0, ptm0o[:,:,64], "t65_img0")
# dt.drawtria(img0, ptm0o[:,:,65], "t66_img0")

