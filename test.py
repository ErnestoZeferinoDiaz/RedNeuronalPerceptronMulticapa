from libs.red import *
from libs.functions import *
import csv
import matplotlib.pyplot as plt

directory = open("in_out_paths/pathsInputs.txt", "r").read().split("\n")

#path = "../train/20.jpg"
#img  = cv2.imread(path)
url = "https://i1.wp.com/www.sopitas.com/wp-content/uploads/2019/09/todos-detalles-elementos-nuevo-billete-200-pesos-destacada-1.png"
img  = url_to_image(url)

X = preprocessingIMG(img)
X = X[0].getA1()
X = np.matrix(X)
Y = np.matrix(np.load("Y.npy"))

pathSave= open("in_out_paths/pathSave.txt", "r").read().split("\n")[0]
params=[]
with open('in_out_paths/config.txt', mode='r') as filee:
   text = csv.reader(filee, delimiter=',')
   for row in text:
      params.append(row)

capas = [int(i) for i in params[0]]
capas.insert(0,X.shape[1])
capas.append(Y.shape[1])

functionesActivacion=params[1]

r = RedNeuronal(
   capas,
   functionesActivacion
)

r.set_X(X)
r.loadModel(pathSave)
y=r.frontPropagation()
m=y.max()
i=np.where(y==m)

print(y)
print(directory[i[0][0]])

#Definimos una lista con los billetes
billetes = ['20', '50', '100', '200', '500', '1000']

#La Lista de las predicciones
prediccion = y.transpose().getA1()
 
fig, ax = plt.subplots()

#Colocamos una etiqueta en el eje Y
ax.set_ylabel('Procentaje')

#Colocamos una etiqueta en el eje X
ax.set_title('Billete')

#Creamos la grafica de barras utilizando 'billetes' como eje X y 'prediccion' como eje y.
plt.bar(billetes, prediccion)

#Finalmente mostramos la grafica con el metodo show()
plt.show()