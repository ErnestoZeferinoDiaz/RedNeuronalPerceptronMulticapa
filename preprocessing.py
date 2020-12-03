from libs.red import *
from libs.functions import *

# leemos la lista de directorios de donde obtendremos las imagenes
directory = open("in_out_paths/pathsInputs.txt", "r").read().split("\n")

X=[]
Y=[]

#recorremos todas las carpetas
for idx_path,path in enumerate(directory):
    print(path)    
    #leemos todos las imagenes de cada carpeta
    for path_img in ls(path):
        # leemos una imagen en especifico
        img  = cv2.imread(path_img)

        # preprocesamos esa imagen que nos da una lista de 4 matrices
        resps = preprocessingIMG(img)

        # hacemos un arreglo de 6 ceros
        tmp = np.zeros(len(directory))
        #actualizamos el indice a uno, depende en que carpeta estemos
        tmp[idx_path]=1

        #recorremos el arreglo de matrices
        for resp in resps:
            # convertimos esa matriz en un arreglo y lo guardamos en X
            X.append(resp.getA1())
            # Guardamos su valor correspondiente en Y
            Y.append(tmp)
        
#Convertimos en matrices 
X = np.matrix(X)
Y = np.matrix(Y)

print()
print(X.shape)
print(Y.shape)

# Guardamos nuestras imagenes y salidas ya preprocesadas
np.save("X",X)
np.save("Y",Y)