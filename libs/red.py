from libs.actFunctions import *
import os

# CLase RedNeuronal que en el constructor recibe
# capas: Un arreglo con la topologia de la red neuronal.
# functions: un arreglo con el string del nombre de las funciones de activacion que se usaran en cada capa
class RedNeuronal:    
    def __init__(self,capas,functions):
        self.fun= functions #arreglo de funciones
        self.capas = capas # arreglo para la topologia de la red
        self.noCapas = len(capas) # largo de la topologia
        self.l = np.array(list(range(self.noCapas))) 
        self.W = []
        self.B = []
        self.Z = []
        self.A = []
        self.S = []
        self.er = 0

    def set_Y(self,Y):
        self.Y = Y

    def set_X(self,X):
        self.X = X
    
    def set_ferror(self,ferror):
        self.ferror= ferror

    def set_alpha(self,alpha):
        self.alpha= alpha

    def reset(self):
        self.W = []
        self.B = []
        self.A = []
        self.S = []              

    def randomModel(self,minimo,maximo):  
        # recorremos el largo de la topologia e inicializamos nuestras matrices de forma aleatoria
        for i in self.l[:-1]:
            self.W.append(minimo + np.random.rand(self.capas[i+1],self.capas[i]) * (maximo - minimo))
            self.B.append(minimo + np.random.rand(self.capas[i+1],1) * (maximo - minimo))

    def frontPropagation(self):
        self.A = []
        # agregamos nuestras entradas
        self.A.append(self.X.transpose()) 

        #recorremos la red neuronal
        for i in self.l[:-1]:
            # Multiplicamos la matriz W por la entrada y le sumamos la matriz B
            z=np.dot(self.W[i],self.A[i]) + self.B[i]

            # le aplicamos la funcion de activacion
            f=mapa[self.fun[i]]
            a=f(z)[0]

            #guardamos la respuesta
            self.A.append(a)

        # Retornamos la ultima salida que es la salida de la red neuronal
        return self.A[-1]
    
    def error(self):
        # calculamos el error
        return self.ferror(self.Y,self.A[-1])[0]

    def backPropagation(self):
        self.S=[]
        
        # recorremos la red de atras hacia adelante
        for i in np.flip(self.l[:-1]): 
            f=mapa[self.fun[i]] #igualamos f a la funcion de activacion
            tmp1 = f(self.A[i+1])[1] #obtenemos la derivada de la funcion

            # calculamos el error de la ultima capa
            if (i==self.noCapas-2):
                tmp2 = self.ferror(self.Y,self.A[-1])[1] # derivada de la funcion de error
                tmp3 = np.multiply(tmp1,tmp2) # multiplicamos por la derivada de la funcion de activacion
                self.S.insert(0,tmp3) # guardamos el error
            else: # calculamos el error en el resto de las capas
                tmp2 = self.S[0] # obtenemos el error de la capa siguiente
                tmp3 = self.W[i+1] # obtenemos los paramentros de la capa siguiente
                tmp4 = np.dot(tmp3.transpose(),tmp2) # multiplicamos los parametros siguientes por el error de la capa
                tmp5 = np.multiply(tmp1,tmp4) # y multiplicamos por la derivada de la funcion de activacion
                self.S.insert(0,tmp5) # guardamos el error
    
    def update(self):
        # recorremos la red
        for i in self.l[:-1]:
            # Hacemos promedio del error para el parametro B
            Sm=np.mean(self.S[i],axis=1)

            # actualizamos valores
            self.W[i] = self.W[i] - self.alpha*np.dot(self.S[i],self.A[i].transpose())
            self.B[i] = self.B[i] - self.alpha*Sm


    def saveModel(self,path):
        if(not os.path.isdir(path)):
            os.mkdir(path)
        
        j=0
        while(j<self.noCapas-1):
            np.save(path+"/W"+str(j),self.W[j])
            np.save(path+"/B"+str(j),self.B[j])
            j = j+1
    
    def loadModel(self,path):
        self.reset()
        for i in range(self.noCapas):        
            self.A.append(0)
            if(i<self.noCapas-1):
                self.W.append(np.matrix(np.load(path+"/W"+str(i)+".npy")))
                self.B.append(np.matrix(np.load(path+"/B"+str(i)+".npy")))
                self.S.append(0)
                