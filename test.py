from libs.red import *
from libs.functions import *
from configPlane import *
import csv

def line(W,B,x):
   i=W[0,0]/W[0,1] * x
   j=B[0,0]/W[0,1]
   return -i-j


X = np.matrix(np.load("X.npy"))
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
W,B=r.loadModel(pathSave)
y=r.frontPropagation()


sep=1
xRange=np.arange(-1,2 + sep,sep)
yRange=np.arange(-1,2 + sep,sep)

f1 = np.matrix([
   [xRange[0],line(W[-1],B[-1],xRange[0])],
   [xRange[-1],line(W[-1],B[-1],xRange[-1])]
])

axs2D[0].plot(f1[:,0],f1[:,1], linestyle = '-', color="red", linewidth=3)

axs2D[0].set_xlim(xRange[0],xRange[-1])
axs2D[0].set_ylim(yRange[0],yRange[-1])
axs2D[0].set_xticks(xRange)
axs2D[0].set_yticks(yRange)

tmp = np.where((Y[:,0]==0))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="blue", marker="o", markersize=10)

tmp = np.where((Y[:,0]==1))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="red",  marker="o", markersize=10)



res=200
u = np.linspace(-1, 2, res)
v = np.linspace(-1, 2, res)
Xt=[]
for i,v1 in enumerate(u):
  for j,v2 in enumerate(v):
    Xt.append([v2,v1])

Xt = np.matrix(Xt)
r.set_X(Xt)
r.loadModel(pathSave)
y=r.frontPropagation().reshape(res,res)

axs2D[0].contourf(u,v,y,alpha=0.5, linewidth=0)

u,v = np.meshgrid(u,v)
axs3D[0].plot_surface(u,v,y,cmap=cm.viridis, alpha=0.5, linewidth=0)

zer=np.zeros((4,1))
tmp = np.where((Y[:,0]==0))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1],zer[tmp], linestyle = 'None', color="blue", marker="o", s=10)

tmp = np.where((Y[:,0]==1))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1],zer[tmp], linestyle = 'None', color="red", marker="o", s=10)
plt.show()