from libs.actFunctions import *
import os
class RedNeuronal:    
    def __init__(self,X,capas,functions):
        self.capas = capas
        self.noCapas = len(capas)
        self.l = np.array(list(range(self.noCapas)))
        self.X = X
        self.W = []
        self.B = []
        self.Z = []
        self.A = []
        self.S = []
        self.fun= functions
        self.er = 0

    def set_Y(self,Y):
        self.Y = Y
    
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
        nR = self.X.shape[0]  
        for i in self.l[:-1]:
            self.W.append(minimo + np.random.rand(self.capas[i+1],self.capas[i]) * (maximo - minimo))
            self.B.append(minimo + np.random.rand(self.capas[i+1],1) * (maximo - minimo))

    def frontPropagation(self):
        self.A = []
        self.A.append(self.X.transpose()) 
        for i in self.l[:-1]:
            z=np.dot(self.W[i],self.A[i]) + self.B[i]
            f=mapa[self.fun[i]]
            a=f(z)[0]
            self.A.append(a)
        return self.A[-1]
    
    def error(self):
        return self.ferror(self.Y,self.A[-1])[0]

    def backPropagation(self):
        self.S=[]
        for i in np.flip(self.l[:-1]): 
            f=mapa[self.fun[i]] 
            tmp1 = f(self.A[i+1])[1]
            if (i==self.noCapas-2):
                tmp2 = self.ferror(self.Y,self.A[-1])[1]    
                tmp3 = np.multiply(tmp1,tmp2)
                self.S.insert(0,tmp3)
            else:
                tmp2 = self.S[0]
                tmp3 = self.W[i+1]
                tmp4 = np.dot(tmp3.transpose(),tmp2)
                tmp5 = np.multiply(tmp1,tmp4)
                self.S.insert(0,tmp5)
    
    def update(self):
        
        for i in self.l[:-1]:
            Sm=np.mean(self.S[i],axis=1)
            #Sm=self.S[i]
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
                