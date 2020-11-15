import numpy as np
import cv2
import os
from os.path import isfile, join

def get_square(image,square_size):
    height,width,alt=image.shape
    if(height>width):
      differ=height
    else:
      differ=width

    mask = np.ones((differ,differ,3), dtype="uint8")*255
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)

    return mask


def ls(ruta = '.'):
    return [ruta+"/"+arch for arch in os.listdir(ruta) if isfile(join(ruta, arch))]