from libs.libs import *
from libs.functions import *

X = np.matrix(np.load("X.npy"))
Y = np.matrix(np.load("Y.npy"))

#print(X.shape)
#print(Y)

rows = X.shape[0]
r = RedNeuronal(
   [X.shape[1],100,50,Y.shape[1]],
   [sigmoid,sigmoid,sigmoid],
   0.01
)

r.reset()
r.randomModel(-1,1)
#r.loadModel("checkpoints")
emedio=[]
eI=1
epocas=0
error = 10**(-2)
while(eI>error):
   suma=0
   for ind,x in enumerate(X):
      r.frontPropagation(x)
      e = r.error(Y[ind])
      r.backPropagation()
      r.update()
      suma = e.transpose()*e + suma
   emedio.append((suma/rows)[0,0]) 
   eI=emedio[epocas]
   epocas=epocas+1
   if(epocas%20==0):
      r.saveModel("checkpoints")
   print(epocas,": ",eI)

