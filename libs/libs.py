import numpy as np
import os

def lineal(x):
    return [x,np.ones(x.shape)]

def relu(x):
    y = np.abs(x)
    z = (y + x)/2
    return [z,np.where(z>0,1,0)]

def sigmoid(x):
    #return [1.0/(1.0+np.exp(-x)),x-np.power(x,2)]
    return [1.0/(1.0+np.exp(-x)),x-np.power(x,2)]

def tanh(x):
    ep=np.exp(x)
    en=np.exp(-x)
    return [(ep-en)/(ep+en),1-np.power(x,2)]

def rectif(x):
    return [np.log(1+np.exp(x)),1.0/(1.0+np.exp(-x))]

def gauss(x):
    a=2
    b=2
    c=2
    f=a*np.exp(-np.power((x-b),2)/(2*c**2))
    return [f,f*5]


class RedNeuronal:
    
    def __init__(self,capas,functions,alpha):
        self.capas = capas
        self.noCapas = len(capas)
        self.Ws = []
        self.Bs = []
        self.Zs = []
        self.As = []
        self.Ss = []
        self.fun= functions
        self.er = 0
        self.alpha = alpha

    def reset(self):
        self.Ws = []
        self.Bs = []
        self.Zs = []
        self.As = []
        self.Ss = []  
             

    def randomModel(self,minimo,maximo):    
        for i in range(self.noCapas):
            self.As.append(0)
            if(i<self.noCapas-1):
                self.Ws.append(minimo + np.random.rand(self.capas[i+1],self.capas[i]) * (maximo - minimo))
                self.Bs.append(minimo + np.random.rand(self.capas[i+1],1) * (maximo - minimo))
                self.Zs.append(0)
                self.Ss.append(0)

    def frontPropagation(self,row):
        self.As[0] = np.matrix(row).transpose()
        j=0
        while(j<self.noCapas-1):
            self.Zs[j] = np.dot(self.Ws[j],self.As[j]) + self.Bs[j]
            self.As[j+1] = self.fun[j](self.Zs[j])[0]                
            j = j+1
        
        return self.As[j]
    
    def error(self,row):
        y = np.matrix(row).transpose()
        self.e = y - self.As[-1]
        return self.e

    def backPropagation(self):
        j=self.noCapas-2
        while(j>=0):            
            if(j==self.noCapas-2):
                self.Ss[j] = -2*np.multiply(self.fun[j](self.As[j+1])[1],self.e)
            else:
                tmp1 = self.fun[j](self.As[j+1])[1]
                self.Ss[j] = np.diagflat(tmp1)*self.Ws[j+1].transpose()*self.Ss[j+1]
            j = j-1
    
    def update(self):
        j=0
        while(j<self.noCapas-1):                
            self.Ws[j] = self.Ws[j] - self.alpha*self.Ss[j]*self.As[j].transpose()
            self.Bs[j] = self.Bs[j] - self.alpha*self.Ss[j]        
            j = j+1
    
    def saveModel(self,path):
        if(not os.path.isdir(path)):
            os.mkdir(path)
        
        j=0
        while(j<self.noCapas-1):
            np.save(path+"/W"+str(j),self.Ws[j])
            np.save(path+"/B"+str(j),self.Bs[j])
            j = j+1
    
    def loadModel(self,path):
        self.reset()
        for i in range(self.noCapas):        
            self.As.append(0)
            if(i<self.noCapas-1):
                self.Ws.append(np.matrix(np.load(path+"/W"+str(i)+".npy")))
                self.Bs.append(np.matrix(np.load(path+"/B"+str(i)+".npy")))
                self.Zs.append(0)
                self.Ss.append(0)
                