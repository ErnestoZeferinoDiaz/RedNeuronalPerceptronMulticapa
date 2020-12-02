from libs.libs import *
from libs.functions import *
import csv

directory = open("in_out_paths/pathsInputs.txt", "r").read().split("\n")

#path = "../train/20.jpg"
#img  = cv2.imread(path)
url = "https://http2.mlstatic.com/D_NQ_NP_755149-MLM26805385609_022018-V.jpg"
img  = url_to_image(url)

X = preprocessingIMG(img)
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