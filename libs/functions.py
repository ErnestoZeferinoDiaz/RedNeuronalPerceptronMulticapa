import numpy as np
import cv2
import os
import urllib.request
from os.path import isfile, join

def preprocessingIMG(img):
    resized = get_square(img,300)
    resp1 = cv2.Canny(resized,50,200)  
    resp2 = cv2.rotate(resp1, cv2.ROTATE_90_CLOCKWISE)
    resp3 = cv2.rotate(resp2, cv2.ROTATE_90_CLOCKWISE)
    resp4 = cv2.rotate(resp3, cv2.ROTATE_90_CLOCKWISE)
    resp = [
      np.matrix(resp1),
      np.matrix(resp2),
      np.matrix(resp3),
      np.matrix(resp4)
    ]
    return resp

def get_square(image,square_size):
  alt    = len(image.shape)
  if(alt==2):
    image = image[..., np.newaxis]
  height = image.shape[0]
  width  = image.shape[1]
  dime   = image.shape[2]
  
  if(height>width):
    differ=height
  else:
    differ=width

  mask = np.ones((differ,differ,dime), dtype="uint8")*255
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