from libs.red import *
import csv, json

config = json.load(open('config.json'))

# Leemos las entradas ya preprocesadas ejecutando normalizar.py
pathInDataX = config["pathData"]["normalizedX"]
pathInDataY = config["pathData"]["normalizedY"]
pathModel   = config["perceptron"]["pathModel"]


# declaramos una funcion de error donde retorne esa funcion y su derivada en un arreglo
def ferror(y,a):
   e=y.transpose()-a
   e2=np.power(e,2)
   de=-2*e
   return [e2,de]

# Leemos los datos preprocesados
X = np.matrix(np.load(pathInDataX+".npy"))
Y = np.matrix(np.load(pathInDataY+".npy"))

# Leemos la ruta en donde se guardara el modelo de la red neuronal
pathSave= pathModel

#Casteamos los parametros en arreglos
capas = [int(i) for i in config["perceptron"]["layers"]]
capas.insert(0,X.shape[1])
capas.append(Y.shape[1])
functionsActivation=config["perceptron"]["activationFunctions"]
alpha=float(config["perceptron"]["alpha"])

# inicializamos red neuronal con los parametros necesarios
r = RedNeuronal(
   capas,
   functionsActivation
)

# le pasamos la funcion de error y el alpha
r.set_ferror(ferror)
r.set_alpha(alpha)

# reseteamos todo lo que pudiera tener la red
r.reset()


# Inicialozamos nuestros parametros de forma aleatoria
# O podemos leer un modelo ya entrenado
isLoadModel = bool(config["perceptron"]["loadingModelTraining"])
if(isLoadModel):
   r.loadModel(pathSave)
else:
   r.randomModel(0,1)

emedio=[]
eI=1
epocas=0
error = config["perceptron"]["errorTrain"]

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
   if(epocas%int(config["perceptron"]["frequencyEpochsSaveModel"])==0):
      print("Guardando...")
      r.saveModel(pathSave)
      print("Listo")

# Si la red llega a su minimo error, volveremos a guardar los datos
r.saveModel(pathSave)