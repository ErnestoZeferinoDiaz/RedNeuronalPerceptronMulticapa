from libs.red import *
from libs.functions import *
import csv

# declaramos una funcion de error donde retorne esa funcion y su derivada en un arreglo
def ferror(y,a):
   e=y.transpose()-a
   e2=np.power(e,2)
   de=-2*e
   return [e2,de]

# Leemos los datos preprocesados
X = np.matrix(np.load("X.npy"))
Y = np.matrix(np.load("Y.npy"))

# Leemos la ruta en donde se guardara el modelo de la red neuronal
pathSave= open("in_out_paths/pathSave.txt", "r").read().split("\n")[0]

#Leemos los parametros de nuestra red neuronal
params=[]
with open('in_out_paths/config.txt', mode='r') as filee:
   text = csv.reader(filee, delimiter=',')
   for row in text:
      params.append(row)

#Casteamos los parametros en arreglos
capas = [int(i) for i in params[0]]
capas.insert(0,X.shape[1])
capas.append(Y.shape[1])
functionesActivacion=params[1]
alpha=float(params[2][0])

# inicializamos red neuronal con los parametros necesarios
r = RedNeuronal(
   capas,
   functionesActivacion
)

# le pasamos la funcion de error y el alpha
r.set_ferror(ferror)
r.set_alpha(alpha)

# reseteamos todo lo que pudiera tener la red
r.reset()

# Inicialozamos nuestros parametros de forma aleatoria
r.randomModel(-1,1)

# O podemos leer un modelo ya entrenado
#r.loadModel(pathSave)


emedio=[]
eI=1
epocas=0
error = 10**(-4)

# Le pasamos a la red, las imagenes con sus respuestas
r.set_X(X)
r.set_Y(Y)

#Repetimos el algoritmo hasta que el error sea minimo
while(eI>error):
   # Propagacion hacia adelante
   a=r.frontPropagation()
   
   #Calculo del error
   e = r.error()

   # Hacemos promedio para obtener un unico valor
   ns=np.sum(e,axis=0)
   nm=np.mean(ns)
   eI=nm
   
   #Hacemos propagacion hacia atras
   r.backPropagation()

   # Actualizamos valores.
   r.update()

   # Contador para indicar en que iteracion vamos
   epocas=epocas+1
   print(epocas,": ",eI)

   # Cada 20 iteraciones guardaremos los datos del entrenamiento
   if(epocas%20==0):
      r.saveModel(pathSave)

# Si la red llega a su minimo error, volveremos a guardar los datos
r.saveModel(pathSave)