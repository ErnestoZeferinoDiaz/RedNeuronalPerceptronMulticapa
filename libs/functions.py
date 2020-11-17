import numpy as np
import cv2
import os
import urllib.request
from os.path import isfile, join

def preprocessingIMG(img):
    resized = get_square(img,300)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  
    border  = cv2.Canny(gray,100,200)  
    resp    = np.matrix(border)
    return resp.getA1()

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

def imgRGBToInt(img):
  w,h,_ = img.shape
  mat = np.zeros((w,h))

  for i in range(w):
    for j in range(h):
      red   = img[i,j,0]
      green = img[i,j,1]
      blue  = img[i,j,2]
      RGBint = (red<<16) + (green<<8) + blue
      mat[i,j] = RGBint
  return mat

def normalize(matrix):
  return matrix * (1/16777215)

def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image