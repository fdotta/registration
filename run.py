import cv2
import numpy as np
import registration as ro
import time

t0 = time.time()
t00 = t0
img0 = ro.readimg( './light/DSC_4364.tiff.fix.tiff')
img1 = ro.readimg( './light/DSC_4384.tiff.fix.tiff')
kp0, des0 = ro.detector(img0)
kp1, des1 = ro.detector(img1)
print "time: %5.2f [kp, des]" % (time.time() - t0)

t0 = time.time()
ptf0, vf0, af0, minarea = ro.ftria1(kp0)
ptf1, vf1, af1 = ro.ftria2(kp1, minarea)
ptm0, ptm1, nf = ro.ftriam(ptf0, ptf1, vf0, vf1, af0, af1, maxmatch=50, err=0.0015)
print "time: %5.2f [match]" % (time.time() - t0)

idx = 15
pp = [ptm1[1,0,idx], ptm1[1,1, idx], 1.]
for i in np.arange(50):
    T = ro.faffine(np.asarray(ptm1[:, :, i]), np.asarray(ptm0[:, :, i]))
    print "real:", ptm0[1,:,idx], "     Transformed:", np.dot(T, pp)
print "Total time %5.2f" % (time.time() - t00)

# pimg0 = Image.fromarray(img0, mode='BRG')
# pimg0 = Image.fromarray(img0, mode='BGR')
# pimg0 = Image.fromarray(img0, mode='BGR')
# pimg0 = Image.fromarray(img0, mode='RGB')
# pimg1 = Image.fromarray(img1, mode='RGB')
# pimg0.show
# pimg0.show()
# draw0 = ImageDraw.Draw(pimg0)
# draw1 = ImageDraw.Draw(pimg1)
# draw0.line((ptm0[0,0,:15]<Plug>PeepOpentm0[0,1,:15], ptm0[1,0,:15]<Plug>PeepOpentm0[1,1,:15]), fill=128)
# draw0.line((ptm0[0, 0,15]<Plug>PeepOpentm0[0,1,15], ptm0[1,0,15]<Plug>PeepOpentm0[1,1,15]), fill=128)
# draw0?
# pimg0.show()
# draw0.line((ptm0[0, 1,15]<Plug>PeepOpentm0[0,1,15], ptm0[1,2,15]<Plug>PeepOpentm0[1,2,15]), fill=128)
# draw0.line((ptm0[0, 0,15], ptm0[0, 0, 15], ptm0[1, 1,15]<Plug>PeepOpentm0[1, 1, 15]), fill=128)
# draw0.line((ptm0[1, 0,15], ptm0[1, 0, 15], ptm0[2, 1,15]<Plug>PeepOpentm0[2, 1, 15]), fill=128)
# draw0.line((ptm0[2, 0,15], ptm0[2, 0, 15], ptm0[0, 1,15]<Plug>PeepOpentm0[0, 1, 15]), fill=128)
# pimg0.show()
# pimg0 = Image.fromarray(img0, mode='RGB')
# draw0 = ImageDraw.Draw(pimg0)
# pimg0.show()
# draw0.line((ptm0[0, 0,15], ptm0[0, 0, 15], ptm0[1, 1,15]<Plug>PeepOpentm0[1, 1, 15]), fill=128)
# pimg0 = Image.fromarray(img0, mode='RGB')
# draw0 = ImageDraw.Draw(pimg0)
# draw0.line((ptm0[0, 0, 15], ptm0[0, 1, 15], ptm0[1, 0, 15]<Plug>PeepOpentm0[1, 1, 15]), fill=128)
# draw0.line((ptm0[1, 0, 15], ptm0[1, 1, 15], ptm0[2, 0, 15]<Plug>PeepOpentm0[2, 1, 15]), fill=128)
# draw0.line((ptm0[2, 0, 15], ptm0[2, 1, 15], ptm0[0, 0, 15]<Plug>PeepOpentm0[0, 1, 15]), fill=128)
# pimg0.show()
# pimg0.save('teste1.jpg')
# draw1.line((ptm1[0, 0, 15], ptm0[0, 1, 15], ptm1[1, 0, 15]<Plug>PeepOpentm0[1, 1, 15]), fill=128)
# draw1.line((ptm1[1, 0, 15], ptm0[1, 1, 15], ptm1[2, 0, 15]<Plug>PeepOpentm0[2, 1, 15]), fill=128)
# draw1.line((ptm1[2, 0, 15], ptm1[2, 1, 15], ptm1[0, 0, 15]<Plug>PeepOpentm1[0, 1, 15]), fill=128)
# pimg1 = Image.fromarray(img1, mode='RGB')
# draw1 = ImageDraw.Draw(pimg1)
# draw1.line((ptm1[0, 0, 15], ptm1[0, 1, 15], ptm1[1, 0, 15]<Plug>PeepOpentm1[1, 1, 15]), fill=128)
# draw1.line((ptm1[1, 0, 15], ptm1[1, 1, 15], ptm1[2, 0, 15]<Plug>PeepOpentm1[2, 1, 15]), fill=128)
# draw1.line((ptm1[2, 0, 15], ptm1[2, 1, 15], ptm1[0, 0, 15]<Plug>PeepOpentm1[0, 1, 15]), fill=128)
# pimg1.show()
