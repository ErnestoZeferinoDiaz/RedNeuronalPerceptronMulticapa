from libs.red import *
from libs.functions import *

X=np.genfromtxt('X.csv', delimiter=',')
Y=np.genfromtxt('Y.csv', delimiter=',')[:,np.newaxis]

print()
print(X.shape)
print(Y.shape)

# Guardamos nuestras imagenes y salidas ya preprocesadas
np.save("X",X)
np.save("Y",Y)