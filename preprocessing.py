from libs.libs import *
from libs.functions import *

directory = open("paths.txt", "r").read().split("\n")

X=[]
Y=[]
for idx_path,path in enumerate(directory):
    print(path)    
    for path_img in ls(path)[:10]:
        img     = cv2.imread(path_img)
        resized = get_square(img,60)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray    = np.matrix(gray)
        X.append(gray.getA1())
        
        tmp = np.zeros(len(directory))
        tmp[idx_path]=1
        Y.append(tmp)

X = np.matrix(X)
Y = np.matrix(Y)

np.save("X",X)
np.save("Y",Y)