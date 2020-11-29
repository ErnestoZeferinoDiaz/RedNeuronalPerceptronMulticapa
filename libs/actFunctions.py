import numpy as np

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

mapa={
  "lineal" :lineal,
  "relu"   :relu,
  "sigmoid":sigmoid,
  "tanh"   :tanh,
  "rectif" :rectif,
  "gauss"  :gauss
}