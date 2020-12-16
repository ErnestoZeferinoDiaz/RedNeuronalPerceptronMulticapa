import numpy as np

# Aqui se encuentran las funciones de activacion
# cada metodo retorna un arreglo.
# este arreglo tiene la funcion de activacion y la derivada de la funcion

def logisticGeneral(k,a,b,x):
  f  = a + ((k-a)/(1+np.exp(-b*x)))

  tmp1 = b*(x-a)
  aux1 = (x-a)
  aux2 = (1/(k-a))  
  tmp2 = aux1*aux2
  tmp3 = 1-tmp2
  df   = np.multiply(tmp1,tmp3)
  
  return [f,df]

def scalon(x):
  f=logisticGeneral(1,0,50,x)[0]
  d=logisticGeneral(1,0,50,x)[1]
  return [f,df]

def lineal(x):
  return [x,np.ones(x.shape)]

def relu(x):
  y = np.abs(x)
  z = (y + x)/2
  return [z,np.where(z>0,1,0)]

def sigmoid(x):
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

# Diccionario que relaciona cada metodo con un string
mapa={
  "scalon" :scalon,
  "lineal" :lineal,
  "relu"   :relu,
  "sigmoid":sigmoid,
  "tanh"   :tanh,
  "rectif" :rectif,
  "gauss"  :gauss
}