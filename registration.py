#/usr/bin/python
"""
Registration
============

Provide
    1. Image reding using CV2
    2. Features detector and descriptor extractor using diferent methods
    3. Draw keypoints in a grayscale image and write on a file
    4. Match the keypoits between 2 images
    5. Determinates the transformation matrix
    6. Transform (warp) an image

Howto use
----------
>>> import registration as ro

"""
import numpy as np
import cv2
import os

def readimg (fname):
    """
    Read image using CV2 and return a numpy RGB array

    Parameters
    ----------
    fname : string
        file name, relative or absolute path

    Returns
    -------
    img : numpy.array
        RGB numpy.array with m x n x 3 dimensions

    Example
    -------
    >>> img = readimg('picture.tiff')
    """
    # print fname
    img = cv2.imread(fname)
    return img

def detector (img, mfeat='STAR', mcomp='BRIEF'):
    """
    Main feature detector and descriptor extractor for image using CV2.

    Parameters
    ----------
    img : numpy.array
        RGB numpy.array with m x n x 3 dimensions
    feat : string, optional
        The desired feature (keypoints) identifier method
    mcomp : string, optional
        The desired descriptor extractor method

    Returns
    -------
    kp : list
        List of the keypoints class with dimension n, where n is the total number
        of keypoinys found
    des : numpy.array
        Array of the descritors of the keypoints with n x m dimensions, where n
        ins the total number of keypoints found

    Feature Indentifer Methods
    --------------------------
        1. `ORB`: Oriented FAST and Rotated BRIEF
        2. `BRISK`: Binary Robust Invariant Scalable Keypoints
        3. `STAR`: StarDetector base on CenSurE::


    Descriptor Extractor Methods
    ----------------------------
        1. `ORB`: Oriented FAST and Rotated BRIEF
        2. `BRISK`: Binary Robust Invariant Scalable Keypoints
        3. `BRIEF`: Binary Robust Independent Elementary Features

    Example
    -------
    >>> kp, des = ro.detector(img)

    >>> kp, des = ro.detector(img, mfeat='ORB')

    >>> kp, des = ro.detector(img, mcomp='ORB')

    >>> kp, des = ro.detector(img, mfeat='STAR', mcomp='ORB')

    """

    if mfeat == 'ORB':
        fmethod = 'ORB'
    elif mfeat == 'BRISK':
        fmethod = 'BRISK'
    elif mfeat == 'STAR':
        fmethod = 'STAR'

    if mcomp == 'ORB':
        dmethod = 'ORB'
    elif mcomp == 'BRIEF':
        dmethod = 'BRIEF'
    elif mcomp == 'BRISK':
        dmethod = 'BRISK'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # initiate keypoints detector
    feature     = cv2.FeatureDetector_create(fmethod)
    # Initiate descriptror extractor
    descriptor  = cv2.DescriptorExtractor_create(dmethod)
    # find the keypoints with STAR
    kp = feature.detect(gray,None)
    # compute the descriptors
    kp, des = descriptor.compute(gray, kp)

    return kp, np.float32(des)

def drawkeypts (img, kp, fname):
    """
    Draw keypoints (without descriptor) in the grayscale image and
    write it in a file

    Parameters
    ----------
    img : numpy.array
        RGB numpy.array with m x n x 3 dimensions
    kp : list
        List of the keypoints class with dimension n, where n is the total number
        of keypoinys found
    fname : string
        file name, relative or absolute path. If the fname was givem without
        extentions will be write a jpg image file. The final file name will be
        <fname>_keypoints.<ext>

    Example
    -------
    >>> drawkeypts(img, kp, 'awesome_picture')

    """
    fsplit = os.path.splitext(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.size(fsplit) == 1:
        fname_tiff = fname+'.jpg'
    else:
        fname_tiff = fsplit[0]+'_keypoints.jpg'
    img2 = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(fname_tiff, img2)
    return

def tvectors (p1, p2, p3):
    """
    Calculate the vectors of a triangle defined by p1, p2 and p3

    Parameters
    ----------
    p1, p2, p3 : numpy.array
        Array of x and y coordinated of the point

    Return
    ------
    v1, v2, v3 : array
        Vectors of the triangle

    """
    v1      = p2 - p1
    v2      = p3 - p1
    v3      = p3 - p2
    return v1, v2, v3

def tarea (v1, v2):
    """
    Calculate the area of a triangle defined by vectors v1 and v2

    Parameters
    ----------
    v1, v2, : numpy.array
        Array of defining a vector

    Return
    ------
    area : float64
        The area of the triangle

    """
    area    = np.linalg.norm(np.cross(v1, v2))/2.

    return area

def tangles (v1, v2, v3):
    """
    Calculate de interal angles of triangles

    Parameters
    ----------
    v1, v2, v3 : numpy.array
        Array of defining a vector

    Return
    ------
    theta : numpy.array
        Angles of the triangle

    """
    theta = np.zeros(3)
    theta[0] = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    theta[1] = np.arccos(np.dot(v2, v3)/(np.linalg.norm(v2) * np.linalg.norm(v3)))
    theta[2] = np.pi - theta[0] - theta[1]

    return theta

def ttransform (v1, v2, v3):
    """
    Transform the triangle from R^2 system to triangle space, where x = a/b
    and y = b/c. The a, b and c are the length of the side of the triangle from
    the bigger to smaller

    Parameters
    ----------
    v1, v2, v3 : numpy.array
        Array of defining a vector

    Return
    ------
    tp : array
        triangle point in the triangle space

    """

    edges   = np.sort(np.array([ np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)]))
    xt      = edges[1] / edges[2]
    yt      = edges[0] / edges[2]
    pt     = np.array([xt, yt])

    return edges, pt

def ftria1 (kp,  minarea=900000, maxtria=20000, tol=0.25):
    """
    find the triangles based on the provided keypoins (kp) of image.
    All triangles with area smaller than minarea will be descosidered. Also
    the function try to find the area to fits only a certain number of
    triangles (maxtria) +/- a tolerance.

    Parameters
    ----------
    kp : list
        List of the keypoints class with dimension n
    minarea : integer, optional
        minimum triangle area to be considered
    tol : integer, optional
        Relative tolerance for the number of triangles found
    maxtria : integer, optinal
        Maximum number of triangles to be found

    Returns
    -------
    ptf : numpy.array
        final triangles points found
    vf : numpy.array
        final triangles vectors found
    af : numpy.array
       final triangles areas
    minarea : integer
       final minimum search area

    """
    lkp       = len(kp)
    ptf       = np.zeros([3, 2, 1])
    vf        = np.zeros([3, 2, 1])
    af        = np.zeros([1, 1])
    nf        = 0
    maxtreach = False
    notend    = True
    minarea1 = 0.8*minarea

    pt = np.zeros([lkp, 2])

    for i in np.arange(lkp):
        pt[i,:] = np.asarray(kp[i].pt)

    xmean1 = np.mean(pt[:,0])
    ymean1 = np.mean(pt[:,1])

    while (notend):
        if nf >= (1.0 + tol)*maxtria:
            nf  = 0
            ptf = np.zeros([3, 2, 1])
            vf  = np.zeros([3, 2, 1])
            af  = np.zeros([1, 1])

        maxtreach = False
        for i1 in np.arange(lkp-2):
            if maxtreach:
                break
            for j1 in np.arange(i1+1,lkp-1):
                if maxtreach:
                    break
                for k1 in np.arange(j1+1,lkp):
                    if maxtreach:
                        break
                    p1           = np.asarray(kp[i1].pt)
                    p2           = np.asarray(kp[j1].pt)
                    p3           = np.asarray(kp[k1].pt)
                    q1            = fquad(p1, xmean1, ymean1)
                    q2            = fquad(p2, xmean1, ymean1)
                    q3            = fquad(p3, xmean1, ymean1)
                    if (q1 != q2) and (q1 != q3) and (q2 != q3):
                        if maxtreach:
                            break
                        v1, v2, v3 = tvectors(p1, p2, p3)
                        area1 = tarea(v1, v2)
                        if area1 >= minarea:
                            if nf == 0:
                                ptf[:, :, 0] = np.array([p1, p2, p3])
                                vf[:, :, 0]  = np.array([v1, v2, v3])
                                af[0, 0]     = area1
                            else:
                                ptf = np.dstack((ptf, np.array([p1, p2, p3])))
                                vf = np.dstack((vf, np.array([v1, v2, v3])))
                                af = np.hstack((af, np.array([[area1]])))
                            nf += 1

                    if nf >= maxtria*(1.0+tol):
                        maxtreach = True
                        minarea1 = minarea
                        minarea = minarea*1.25
                        break

                    if (i1 >= (lkp - 3)) and (nf >= maxtria*(1.0 - tol)) and \
                            nf <= maxtria*(1.0+tol):
                        maxtreach = True
                        notend    = False
                        break
                    elif (i1 >= lkp - 3) and (nf <= maxtria*(1.0 - tol)):
                        maxtreach = True
                        minarea = (minarea1 + minarea)/2
                        break

            # print "NTria = %5i Pt1 = %3i Area = %i" % (nf, i1, np.int(minarea))
        if (i1 >= lkp - 2):
            notend = True
    return ptf, vf, af, np.int(minarea)

def ftria2 (kp, minarea):
    """
    find the triangles based on the provided keypoins (kp) of image and minimum
    area of the triangle.

    Parameters
    ----------
    kp : list
        List of the keypoints class with dimension n
    minarea : integer, optional
        minimum triangle area to be considered

    Returns
    -------
    ptf : numpy.array
        final triangles points found
    vf : numpy.array
        final triangles vectors found
    af : numpy.array
       final triangles areas
    """
    lkp       = len(kp)
    ptf       = np.zeros([3, 2, 1])
    vf        = np.zeros([3, 2, 1])
    af        = np.zeros([1, 1])
    nf        = 0
    for i1 in np.arange(lkp-2):
        for j1 in np.arange(i1+1,lkp-1):
            for k1 in np.arange(j1+1,lkp):
                p1           = np.asarray(kp[i1].pt)
                p2           = np.asarray(kp[j1].pt)
                p3           = np.asarray(kp[k1].pt)
                v1, v2, v3 = tvectors(p1, p2, p3)
                area1 = tarea(v1, v2)
                if area1 >= minarea:
                    if nf == 0:
                        ptf[:, :, 0] = np.array([p1, p2, p3])
                        vf[:, :, 0]  = np.array([v1, v2, v3])
                        af[0, 0]     = area1
                    else:
                        ptf = np.dstack((ptf, np.array([p1, p2, p3])))
                        vf = np.dstack((vf, np.array([v1, v2, v3])))
                        af = np.hstack((af, np.array([[area1]])))
                    nf += 1

    return ptf, vf, af

def fquad (p, xmean, ymean):
    """
    find the quadrant where the point is located based on the mean point (xmean,ymean)

    Parameters
    ----------
    p : numpy.array
        array of the point x and y
    xmean : float
        X mean value
    ymean :  float
        Y mean value

    Return
    ------

    quadrant : integer
        Quadrant (1, 2, 3, 4)

    """

    if (p[0] < xmean) and (p[1] > ymean):
        quadrant = 1
    elif (p[0] >= xmean) and (p[1] > ymean):
        quadrant = 2
    elif (p[0] < xmean) and (p[1] <= ymean):
        quadrant = 3
    else:
        quadrant = 4

    return quadrant

def ftriam (ptf1, ptf2, vf1, vf2, af1, af2, maxmatch=120, err=0.001):
    """
    Find similar triangles form ptf1 in ptf2

    Parameters
    ----------
    ptf1, ptf2 : numpy.array
        final triangles points found
    vf1, vf2 : numpy.array
        final triangles vectors found
    af1, af2 : numpy.array
       final triangles areas
    maxmatch : integer, optional
        Maximum triangles to find
    err : float
        relative error

    Returns
    -------
    ptm1, ptm2 : numpy.array
        Matched triangles points
    nf : integer
        Number of triangles matched
    """
    np1  = ptf1.shape[2]
    np2  = ptf2.shape[2]
    nf   = 0
    ptm1 = np.zeros([3, 2, maxmatch])
    ptm2 = np.zeros([3, 2, maxmatch])

    for i1 in np.arange(np1):
        if nf >= maxmatch:
            break
        edges1, pt1 = ttransform(vf1[0,:,i1], vf1[1,:,i1], vf1[2,:,i1])
        theta1 = tangles (vf1[0,:,i1], vf1[1,:,i1], vf1[2,:,i1])
        xerr = err*pt1[0]
        yerr = err*pt1[1]
        terr = err*np.max(theta1)
        aerr = err*af1[0,i1]
        for i2 in np.arange(np2):
            if nf >= maxmatch:
                break
            if (af1[0, i1] - aerr < af2[0, i2]) and (af1[0, i1] + aerr >= af2[0, i2]):
                edges2, pt2 = ttransform(vf2[0, :, i2], vf2[1, :, i2], vf2[2, :, i2])
                theta2 = tangles (vf2[0, :, i2], vf2[1, :, i2], vf2[2, :, i2])
                if (pt1[0] - xerr < pt2[0]) and (pt1[0] + xerr >= pt2[0]):
                    if (pt1[1] - yerr < pt2[1]) and (pt1[1] + yerr >= pt2[1]):
                        tfind = False
                        for j1 in np.arange(3):
                            if tfind:
                                break
                            for j2 in np.arange(3):
                                if tfind:
                                    break
                                ttmp = np.asarray(np.where(theta1 - terr < theta2) and (theta1 + terr >= theta2))
                                if ttmp.shape[0] == 3:
                                    nf += 1
                                    ptm1[:, :, nf-1] = ptf1[:, :, i1]
                                    ptm2[:, :, nf-1] = ptf2[:, :, i2]
                                    # print ">", nf, i2, pt1, pt2, xerr, yerr
                                    # print "   ", theta1, theta2, terr
                                    tfind = True

    return ptm1, ptm2, nf

def faffine (src, dst):
    """
    Parameters
    ----------

    Returns
    -------

    """
    src = np.float32(src)
    dst = np.float32(dst)
    T = cv2.getAffineTransform(src, dst)

    return T

def Tmatrix (ptm1, ptm2):
    nptm         = ptm1.shape[2]
    T            = np.zeros([2, 3, 1])
    pt           = np.zeros([nptm, 2])
    notend       = False
    nopointvalid = False
    tpp          = 0

    for i in np.arange(nptm):
        Tidx = faffine(np.asarray(ptm2[:, :, i]), np.asarray(ptm1[:, :, i]))
        if i == 0:
            T[: , :, i] = Tidx[:, :]
        else:
            T = np.dstack((T, Tidx))

    while (not notend):
        for idx in np.arange(nptm):
            pp = [ptm2[tpp,0,idx], ptm2[tpp,1, idx], 1.]
            for i in np.arange(nptm):
                pt[i, :] =  np.dot(T[:, :, i], pp)
                # print idx, "real:", ptm1[1, :, idx], "     Transformed:", pt[i, :]

            xo = foutliers(pt[:, 0])
            try:
               len(xo)
            except TypeError:
               nopointvalid = True
               continue
            else:
                nopointvalid = False

            xo_rrange = np.arange(nptm)
            xo_rrange = xo_rrange[::-1]

            for i in xo_rrange:
                if not xo[i]:
                    pt = np.delete(pt, i, axis=0)
            if not ((np.mean(pt[0,:]) >=  0.85*ptm1[tpp, 0, idx]) and (np.mean(pt[0,:]) < 1.15*ptm1[tpp, 0, idx])):
                pt = np.zeros([nptm, 2])
            else:
                for i in xo_rrange:
                    if not xo[i]:
                        T    = np.delete(T, i, axis=2)
                        ptm1 = np.delete(ptm1, i, axis=2)
                        ptm2 = np.delete(ptm2, i, axis=2)
                notend = True
                break
        tpp += 1
        if tpp == 3:
            notend       = True
            nopointvalid = True

    if not nopointvalid:
        yo = foutliers(pt[:, 1])

        nptm = ptm1.shape[2]
        yo_rrange = np.arange(nptm)
        yo_rrange = yo_rrange[::-1]
        for i in yo_rrange:
            if not yo[i]:
                T    = np.delete(T, i, axis=2)
                ptm1 = np.delete(ptm1, i, axis=2)
                ptm2 = np.delete(ptm2, i, axis=2)
                pt = np.delete(pt, i, axis=0)

        TM = np.zeros([2, 3])

        for i in np.arange(2):
            for j in np.arange(3):
                TM[i, j] = np.mean(T[i, j, :])
    else:
        TM = np.zeros([2, 3])

    return ptm1, ptm2, TM, not nopointvalid

def foutliers(data, m=2.0):
    """

    """
    ndata = data.shape[0]
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s<m





####### to be handle latter ###############################

# def fmatch (des1, des2, kp1, kp2, ratio=0.700):
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # matches = flann.knnMatch(des1, des2, k=2)

# # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
        # if m.distance < ratio*n.distance:
        #     good.append(m)
    # return good, matches

# def pointmatch (kp1, kp2, good, MIN_MATCH_COUNT = 3):
    # if len(good)>MIN_MATCH_COUNT:
        # pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        # pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # matcher = cv2.BFMatcher(cv2.NORM_L2)

    # matches = matcher.knnMatch(des1, trainDescriptors = des2, k = 2) #2

    # mkp1, mkp2 = [], []
    # for m in matches:
    #     if len(m) == 2 and m[0].distance < m[1].distance * ratio:
    #         m = m[0]
    #         mkp1.append( kp1[m.queryIdx] )
    #         mkp2.append( kp2[m.trainIdx] )
    #         kp_pairs = zip(mkp1, mkp2)
    # return kp_pairs

    # # return pts1, pts2


