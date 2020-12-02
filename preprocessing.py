from libs.red import *
from libs.functions import *


directory = open("in_out_paths/pathsInputs.txt", "r").read().split("\n")

X=[]
Y=[]

for idx_path,path in enumerate(directory):
    print(path)    
    for path_img in ls(path):
        img  = cv2.imread(path_img)
        resp1 = preprocessingIMG(img)
        resp2 = cv2.rotate(resp1, cv2.ROTATE_90_CLOCKWISE)
        resp3 = cv2.rotate(resp2, cv2.ROTATE_90_CLOCKWISE)
        resp4 = cv2.rotate(resp3, cv2.ROTATE_90_CLOCKWISE)

        X.append(resp1.getA1())
        X.append(resp2.getA1())
        X.append(resp3.getA1())
        X.append(resp4.getA1())
        
        tmp = np.zeros(len(directory))
        tmp[idx_path]=1
        Y.append(tmp)
        Y.append(tmp)
        Y.append(tmp)
        Y.append(tmp)

X = np.matrix(X)
Y = np.matrix(Y)

np.save("X",X)
np.save("Y",Y)