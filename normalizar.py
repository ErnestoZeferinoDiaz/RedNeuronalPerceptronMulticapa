import numpy as np


pathInDataRaw         = 'data/raw/dataSet.csv'
pathOutRows           = 'data/support/rows'
pathOutXmin           = 'data/support/Xmin'
pathOutXmax           = 'data/support/Xmax'
pathOutDataNormalized = 'data/normalized/dataSetNorm'
pathOutDataX          = 'data/normalized/X'
pathOutDataY          = 'data/normalized/Y'


# leemos los datos en crudo
data=np.genfromtxt(pathInDataRaw,delimiter=',',dtype="float")

# separamos la matriz en las entradas y en las salidas
X=data[:,0:5]
Y=np.matrix(data[:,5]).transpose() # trasponemos ya que como es una sola columna numpy lo toma como lista 

# vemos cuantos registros tenemos
rows=X.shape[0]

# vector para separar un valor en una lista de 0s y 1s
Yn=np.zeros((rows,27))
for i,v in enumerate(Y):
    Yn[i,int(v[0,0])]=1

#buscamos el valor minimo y maximo de las entradas en cada columna
Xmin=np.amin(X,axis=0)
Xmax=np.amax(X,axis=0)

#normalizamos de la forma X = (X-Xmin)/(Xmax-Xmin)
Xn=X-np.tile(Xmin, (rows, 1))
Xn=Xn/(np.tile(Xmax, (rows, 1))-np.tile(Xmin, (rows, 1)))

#unimos las entradas ya normalizadas y las nuevas salidas
dataN=np.zeros((rows,Xn.shape[1]+Yn.shape[1]))
dataN[:,0:5]=Xn
dataN[:,5:]=Yn

#guardamos todos los datos ya que los necesitaremos para el test
np.save(pathOutDataX,Xn)
np.savetxt(pathOutDataX+".csv",Xn,delimiter=",",fmt="%g")

np.save(pathOutDataY,Yn)
np.savetxt(pathOutDataY+".csv",Yn,delimiter=",",fmt="%g")

np.save(pathOutRows,np.array([rows]))
np.savetxt(pathOutRows+".csv",np.array([rows]),delimiter=",",fmt="%g")

np.save(pathOutXmin,Xmin)
np.savetxt(pathOutXmin+".csv",Xmin,delimiter=",",fmt="%g")

np.save(pathOutXmax,Xmax)
np.savetxt(pathOutXmax+".csv",Xmax,delimiter=",",fmt="%g")

np.save(pathOutDataNormalized,dataN)
np.savetxt(pathOutDataNormalized+".csv",dataN,delimiter=",",fmt="%g")
