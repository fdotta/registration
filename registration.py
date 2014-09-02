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


def ftria1 (kp,  minarea=90000):
    """
    find the similar triangles based on the provided keypoins (kp1 and kp2).
    All triangles with area, a lenght smaller than minarea and mina,
    respectively will be desconsidered. The search will be stoped when the
    minimum findings (nfinds) or end of keypoints was reached.

    Parameters
    ----------
    kp1 : list
        List of the keypoints class with dimension n
    minarea : integer, optional
        minimum triangle area to be considered
    mina : integer, optional
        minimum triangle a lenght of triangle (longest side)
    nfinds : integer, optional
        number of similar triangles target finding

    Return
    ------
    nt1 :  interger
        Number of similar trinagle finded
    tfound : numpy.array
        Array with the points of similar triangles tfound[1:3,1:2,:] -> image 1 (kp1)
        tfound[4:6,1:2,:] -> image 2  (kp2).
        dim 1 -> triangles, dim 2 -> x, y

    """
    lkp = len(kp)
    nf  = 0
    ptf = np.zeros([3, 2, 1])
    tf  = np.zeros([3, 1])
    vf  = np.zeros([3, 2, 1])
    af  = np.zeros([1, 1])

    pt = np.zeros([lkp, 2])

    for i in np.arange(lkp):
        pt[i,:] = np.asarray(kp[i].pt)

    xmean1 = np.mean(pt[:,0])
    ymean1 = np.mean(pt[:,1])

    for i1 in np.arange(lkp):
        for j1 in np.arange(i1+1,lkp):
            for k1 in np.arange(j1+1,lkp):
                p1           = np.asarray(kp[i1].pt)
                p2           = np.asarray(kp[j1].pt)
                p3           = np.asarray(kp[k1].pt)
                q1            = fquad(p1, xmean1, ymean1)
                q2            = fquad(p2, xmean1, ymean1)
                q3            = fquad(p3, xmean1, ymean1)
                if (q1 != q2) and (q1 != q3) and (q2 != q3):
                    # print q1, q2, q3, (q1 != q2), (q1 != q3), (q2 != q3)
                    v1, v2, v3 = tvectors(p1, p2, p3)
                    area1 = tarea(v1, v2)
                    theta = np.asarray(tangles(v1, v2, v3))
                    if area1 >= minarea:
                        if nf == 0:
                            ptf[:, :, 0] = np.array([p1, p2, p3])
                            tf[:, 0]     = theta
                            vf[:, :, 0]  = np.array([v1, v2, v3])
                            af[0, 0]     = area1
                        else:
                            ptf = np.dstack((ptf, np.array([p1, p2, p3])))
                            vf = np.dstack((vf, np.array([v1, v2, v3])))
                            tf = np.hstack((tf, theta.reshape([3, 1])))
                            af = np.hstack((af, np.array([[area1]])))
                        nf += 1
    return ptf, vf, tf, af

def ftria2 (pt, theta, kp, err=0.002):
    """
    find the similar triangle in a set of points

    """
    xerr = err*pt[0]
    yerr = err*pt[1]
    lkp = len(kp)
    for i1 in np.arange(lkp):
        for j1 in np.arange(i1+1,lkp):
            for k1 in np.arange(j1+1,lkp):
                p1         = np.asarray(kp[i1].pt)
                p2         = np.asarray(kp[j1].pt)
                p3         = np.asarray(kp[k1].pt)
                v1, v2, v3 = tvectors(p1, p2, p3)
                edges, pt2 = ttransform (v1, v2, v3)
                if (pt[0] - xerr < pt2[0]) and (pt[0] + xerr >= pt2[0]):
                    if (pt[1] - yerr < pt2[1]) and (pt[1] + yerr >= pt2[1]):
                            print pt, pt2,xerr, yerr

    return



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

####### to be handle latter ###############################

def fmatch (des1, des2, kp1, kp2, ratio=0.700):
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

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(des1, trainDescriptors = des2, k = 2) #2

    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
            kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

    # return pts1, pts2

def faffine (src, dst):

    tria1 = (src[0], src[1], src[2])
    tria2 = (dst[0], dst[1], dst[2])
    T = cv2.getAffineTransform(tria1, tria2)

    return T

