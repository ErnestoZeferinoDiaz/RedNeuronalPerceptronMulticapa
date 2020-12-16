from libs.red import *
from libs.functions import *

X=[
  [0.7,  3],
  [1.5,  6],
  [2,    9],
  [0.9, 11],
  [4.2,  0],
  [2.2,  1],
  [3.6,  7],
  [4.5,  6]
]

Y=[
  [0,0],
  [0,0],
  [0,1],
  [0,1],
  [1,0],
  [1,0],
  [1,1],
  [1,1]
]
        
#Convertimos en matrices 
X = np.matrix(X)
Y = np.matrix(Y)

print()
print(X.shape)
print(Y.shape)

# Guardamos nuestras imagenes y salidas ya preprocesadas
np.save("X",X)
np.save("Y",Y)