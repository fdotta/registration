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
    mfeat : string, optional
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
    area    = np.linalg.norm(np.cross(v1,v2))/2.

    return area

















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

