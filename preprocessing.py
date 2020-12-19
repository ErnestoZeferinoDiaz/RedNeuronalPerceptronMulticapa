from libs.red import *
from libs.functions import *

X=[
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]
Y=[
  [0],
  [1],
  [1],
  [0]
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