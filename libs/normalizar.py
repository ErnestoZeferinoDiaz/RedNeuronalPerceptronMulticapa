import numpy as np

def normalize(dataIn,phase=0,rows=None,Xmin=None,Xmax=None):

    # separamos la matriz en las entradas y en las salidas
    X=dataIn[:,0:5]
    
    if(rows is None):
        # vemos cuantos registros tenemos
        rows=X.shape[0]

    #buscamos el valor minimo y maximo de las entradas en cada columna
    if(Xmin is None):
        Xmin=np.amin(X,axis=0)
    
    if(Xmax is None):
        Xmax=np.amax(X,axis=0)

    #normalizamos de la forma X = (X-Xmin)/(Xmax-Xmin)
    Xn=X-np.tile(Xmin, (rows, 1))
    Xn=Xn/(np.tile(Xmax, (rows, 1))-np.tile(Xmin, (rows, 1)))

    Yn=None
    if(phase==0):
        Y=np.matrix(dataIn[:,5]).transpose() # trasponemos ya que como es una sola columna numpy lo toma como lista 
        # vector para separar un valor en una lista de 0s y 1s
        Yn=np.zeros((rows,27))
        for i,v in enumerate(Y):
            Yn[i,int(v[0,0])]=1


    return [Xn,Yn,rows,Xmin,Xmax]
