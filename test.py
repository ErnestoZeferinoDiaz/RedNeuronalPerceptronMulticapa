from libs.red import *
from libs.functions import *
import csv

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