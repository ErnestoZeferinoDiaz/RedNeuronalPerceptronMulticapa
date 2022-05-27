from libs.red import *
from libs.normalizar import *
from libs.interface import *
from tkinter import *
import csv, serial, time

pathModel   = 'model'
pathConfig  = 'config/config.txt'
pathInRows  = 'data/support/rows'
pathInXmin  = 'data/support/Xmin'
pathInXmax  = 'data/support/Xmax'
pathInDataX = 'data/normalized/X'
pathInDataY = 'data/normalized/Y'

# leemos datos de soporte para la normalizacion de nuestra entrada de datos
rows=np.matrix(np.load(pathInRows+".npy"))
Xmin=np.matrix(np.load(pathInXmin+".npy"))
Xmax=np.matrix(np.load(pathInXmax+".npy"))

#Leemos los parametros de nuestra red neuronal
pathSave= pathModel
params=[]
with open(pathConfig, mode='r') as filee:
   text = csv.reader(filee, delimiter=',')
   for row in text:
      params.append(row)
Xs = np.matrix(np.load(pathInDataX+".npy")).shape[1]
Ys = np.matrix(np.load(pathInDataY+".npy")).shape[1]

#Casteamos los parametros en arreglos
capas = [int(i) for i in params[0]]
capas.insert(0,Xs)
capas.append(Ys)
functionsActivation=params[1]
alpha=float(params[2][0])

# Iniciamos la red neuronal
red = RedNeuronal(
   capas,
   functionsActivation
)

# cargamos el modelo entrenado
red.loadModel(pathSave)

arduino = serial.Serial("COM4",9600,timeout=1.0)
arduino.setDTR(False)
time.sleep(1)
arduino.flushInput()
arduino.setDTR(True)

def funLoop(sel,r):
   # Recibimos la entrada de datos
   rawString = arduino.readline()
   cad=rawString.decode()[:-2]
   listCad=cad.split(",")
   data=list(map(float, listCad))[:-1]
   #data = [[642.00,624.00,654.00,650.00,572.00]]

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