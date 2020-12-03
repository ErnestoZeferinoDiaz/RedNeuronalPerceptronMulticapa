import numpy as np
import cv2
import os
import urllib.request
from os.path import isfile, join

#Funcion que preprocesa una imagen en 4 imagenes
def preprocessingIMG(img):
    #Redimencionamos una imagen a 300x300
    resized = get_square(img,300)

    #Sacamos los bordes de la imagen
    resp1 = cv2.Canny(resized,50,200)  

    # +++ Mostrar una imagen +++
    #cv2.imshow('Imagen bordes',resp1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # La imagen de los bordes la replicamos 3 veces, giradas 90 grados tres veces
    resp2 = cv2.rotate(resp1, cv2.ROTATE_90_CLOCKWISE)
    resp3 = cv2.rotate(resp2, cv2.ROTATE_90_CLOCKWISE)
    resp4 = cv2.rotate(resp3, cv2.ROTATE_90_CLOCKWISE)

    #Convertimos en matrix las 4 imagenes y retornamos un arreglo con esas 4 imagenes
    resp = [
      np.matrix(resp1),
      np.matrix(resp2),
      np.matrix(resp3),
      np.matrix(resp4)
    ]
    return resp

# funcion que redimenciona cualquier imagen de forma cuadrada
def get_square(image,square_size):
  # Si es una imagen en blanco y negro, solo tendra una dimencion ejemplo (500,300)
  # Pero si es a color tendra 3 dimenciones por los tres colores Red-Green-Blue ejemplo (500,300,3)
  # Por lo tanto si es a blanco y negro, agregaremos otra dimencion aunque tenga el valor de 1 es decir (500,300,1)
  alt    = len(image.shape)
  if(alt==2):
    image = image[..., np.newaxis]

  #Ahora obtenemos el largo y ancho de la imagen y si es de una dimencion o 3 dimenciones
  height = image.shape[0]
  width  = image.shape[1]
  dime   = image.shape[2]
  
  #verificamos que lado es mas largo si en largo o el ancho
  if(height>width):
    differ=height
  else:
    differ=width
  
  # Hacemos una imagen cuadrada del lado que sea mas grande de color blanco
  # Si por ejemplo la iamgen de entrada es de 500x600 haremos una imagen de 600x600
  mask = np.ones((differ,differ,dime), dtype="uint8")*255

  # ubicamos desde donde se va a colocar la imagen ya redimencionada en la imagen grande de blanco
  x_pos=int((differ-width)/2)
  y_pos=int((differ-height)/2)

  # Colocamos la imagen original en la imagen grande
  mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]

  # Ahora si redimencionamos la imagen grande al tamaño que deseamos
  mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)
  return mask

# Suponga que en la carpeta misImagenes tiene 3 archivos. hola.txt, mundo.csv, genial.jpg
# Lo retorna la funcion es una lista de rutas dependiendo la cantidad de archivos. En este caso
# [ misImagenes/hola.txt, misImagenes/mundo.csv, misImagenes/genial.jpg ]
def ls(ruta = '.'):
  return [ruta+"/"+arch for arch in os.listdir(ruta) if isfile(join(ruta, arch))]

# Dada una url de una imagen, retorna la imagen
def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image