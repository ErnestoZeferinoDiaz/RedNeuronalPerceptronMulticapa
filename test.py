from libs.red import *
from libs.normalizar import *
from libs.interface import *
from tkinter import *
import csv, serial, time, json

config = json.load(open('config.json'))

pathModel   = config["perceptron"]["pathModel"]
pathInDataX = config["pathData"]["normalizedX"]
pathInDataY = config["pathData"]["normalizedY"]
pathInRows  = config["pathData"]["rows"]
pathInXmin  = config["pathData"]["xmin"]
pathInXmax  = config["pathData"]["xmax"]
port = config["arduino"]["port"]

# leemos datos de soporte para la normalizacion de nuestra entrada de datos
rows=np.matrix(np.load(pathInRows+".npy"))
Xmin=np.matrix(np.load(pathInXmin+".npy"))
Xmax=np.matrix(np.load(pathInXmax+".npy"))

#Leemos los parametros de nuestra red neuronal
pathSave= pathModel

Xs = np.matrix(np.load(pathInDataX+".npy")).shape[1]
Ys = np.matrix(np.load(pathInDataY+".npy")).shape[1]

#Casteamos los parametros en arreglos
capas = [int(i) for i in config["perceptron"]["layers"]]
capas.insert(0,Xs)
capas.append(Ys)
functionsActivation=config["perceptron"]["activationFunctions"]
alpha=float(config["perceptron"]["alpha"])

# Iniciamos la red neuronal
red = RedNeuronal(
   capas,
   functionsActivation
)

# cargamos el modelo entrenado
red.loadModel(pathSave)

arduino = serial.Serial(port,9600,timeout=1.0)
arduino.setDTR(False)
time.sleep(1)
arduino.flushInput()
arduino.setDTR(True)
arduino.readline()
def funLoop(sel,r):
   # Recibimos la entrada de datos
   rawString = arduino.readline()
   cad=rawString.decode()[:-2]
   listCad=cad.split(",")
   data=list(map(float, listCad))
   #data = [[642.00,624.00,654.00,650.00,572.00]]
   print(data)

   # Normalizamos nuestra entrada
   dataIn = np.matrix(data,dtype="float")
   X = normalize(dataIn,1,1,Xmin,Xmax)[0]

   # Le pasamos la entrada ya normalizada
   r.set_X(X)

   # Propagamos la entrada hacia adelante para obtener la salida
   Y=r.frontPropagation()

   # Aplicamos funcion softMax a la salida de la neurona para exponenciar los resultados
   sumExp=np.sum(np.exp(Y))
   softMax=np.exp(Y)/sumExp

   # Obtenemos el indice de la letra, calculando el valor maximo de nuestra salida
   indexChar = np.argmax(softMax)
   
   # Actualizamos letra en la interface
   sel.button1["text"]=sel.abc[indexChar]
   print("funciona")


root = tkinter.Tk()
root.configure(background='#c8c8c8')
root.title("monitor")
root.geometry("700x700")
app = Application(root)
app.set_Function_Loop(lambda se: funLoop(se,red))
app.run()
root.mainloop()