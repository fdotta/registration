#/usr/bin/python
import numpy as np
import cv2
from PIL import Image, ImageDraw

def drawtria(img, pt, name):
    """

    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pimg = Image.fromarray(img, mode='RGB')
    draw = ImageDraw.Draw(pimg)
    draw.line((pt[0, 0], pt[0, 1], pt[1, 0], pt[1, 1]), fill=255,  width=3)
    draw.line((pt[1, 0], pt[1, 1], pt[2, 0], pt[2, 1]), fill=255,  width=3)
    draw.line((pt[2, 0], pt[2, 1], pt[0, 0], pt[0, 1]), fill=255,  width=3)
#    pimg.show()
    fname  = name + ".jpg"
    pimg.save(fname)

