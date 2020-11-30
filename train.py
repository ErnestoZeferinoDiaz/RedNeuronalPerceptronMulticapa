from libs.libs import *
from libs.functions import *
import csv

def ferror(y,a):
   e=y.transpose()-a
   e2=np.power(e,2)
   de=-2*e
   return [e2,de]

X = np.matrix(np.load("X.npy"))
Y = np.matrix(np.load("Y.npy"))

pathSave= open("in_out_paths/pathSave.txt", "r").read().split("\n")[0]
params=[]
with open('in_out_paths/config.csv', mode='r') as filee:
   text = csv.reader(filee, delimiter=',')
   for row in text:
      params.append(row)

capas = [int(i) for i in params[0]]
capas.insert(0,X.shape[1])
capas.append(Y.shape[1])

functionesActivacion=params[1]
alpha=float(params[2][0])

r = RedNeuronal(
   capas,
   functionesActivacion
)

r.set_ferror(ferror)
r.set_alpha(alpha)

r.reset()
r.randomModel(-1,1)
#r.loadModel(pathSave)

emedio=[]
eI=1
epocas=0
error = 10**(-4)

r.set_X(X)
r.set_Y(Y)
while(eI>error):
   a=r.frontPropagation()
   
   e = r.error()
   ns=np.sum(e,axis=0)
   nm=np.mean(ns)
   eI=nm
   
   r.backPropagation()
   r.update()

   epocas=epocas+1
   print(epocas,": ",eI)

   if(epocas%20==0):
      r.saveModel(pathSave)
r.saveModel(pathSave)