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
xRange=np.arange(-1,6 + sep,sep)
yRange=np.arange(-1,12 + sep,sep)

f1 = np.matrix([
   [xRange[0], line(W[-1][0],B[-1][0],xRange[0])],
   [xRange[-1],line(W[-1][0],B[-1][0],xRange[-1])]
])

f2 = np.matrix([
   [xRange[0], line(W[-1][1],B[-1][1],xRange[0])],
   [xRange[-1],line(W[-1][1],B[-1][1],xRange[-1])]
])

axs2D[0].plot(f1[:,0],f1[:,1], linestyle = '-', color="lime", linewidth=3)
axs2D[0].plot(f2[:,0],f2[:,1], linestyle = '-', color="lime", linewidth=3)

axs2D[0].set_xlim(xRange[0],xRange[-1])
axs2D[0].set_ylim(yRange[0],yRange[-1])
axs2D[0].set_xticks(xRange)
axs2D[0].set_yticks(yRange)

tmp = np.where((Y[:,0]==0) & (Y[:,1]==0))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="red", marker="o", markersize=10)

tmp = np.where((Y[:,0]==0) & (Y[:,1]==1))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="red", marker="^", markersize=10)

tmp = np.where((Y[:,0]==1) & (Y[:,1]==0))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="cyan",marker="o", markersize=10)

tmp = np.where((Y[:,0]==1) & (Y[:,1]==1))[0]
axs2D[0].plot(X[tmp,0],X[tmp,1], linestyle = 'None', color="cyan",marker="^", markersize=10)

res=100
u = np.linspace(xRange[0],xRange[-1], res)
v = np.linspace(yRange[0],yRange[-1], res)
Xt=[]
for i,v2 in enumerate(v):
  for j,v1 in enumerate(u):
    Xt.append([v1,v2])

Xt = np.matrix(Xt)
r.set_X(Xt)
r.loadModel(pathSave)
y=r.frontPropagation()

axs2D[0].contourf(u,v,y[0].reshape(res,res),cmap=cm.cool,alpha=0.5, linewidth=0)
axs2D[0].contourf(u,v,y[1].reshape(res,res),cmap=cm.bwr,    alpha=0.5, linewidth=0)

u,v = np.meshgrid(u,v)
axs3D[0].plot_surface(u,v,y[0].reshape(res,res),cmap=cm.cool, alpha=0.5, linewidth=0)
axs3D[0].plot_surface(u,v,y[1].reshape(res,res),cmap=cm.bwr    , alpha=0.5, linewidth=0)

zer=np.zeros((8,1))
one=np.ones((8,1))

tmp = np.where((Y[:,0]==0) & (Y[:,1]==0))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1], zer[tmp],linestyle = 'None', color="red", marker="o", s=50)

tmp = np.where((Y[:,0]==0) & (Y[:,1]==1))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1], zer[tmp],linestyle = 'None', color="red", marker="^", s=50)

tmp = np.where((Y[:,0]==1) & (Y[:,1]==0))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1], zer[tmp],linestyle = 'None', color="cyan",marker="o", s=50)

tmp = np.where((Y[:,0]==1) & (Y[:,1]==1))[0]
axs3D[0].scatter(X[tmp,0],X[tmp,1], zer[tmp],linestyle = 'None', color="cyan",marker="^", s=50)

plt.show()