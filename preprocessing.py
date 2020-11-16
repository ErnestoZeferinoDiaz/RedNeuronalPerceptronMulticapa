from libs.libs import *
from libs.functions import *

def preprocessingIMG(path_img):
    img     = cv2.imread(path_img)
    resized = get_square(img,200)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    borde   = cv2.Canny(gray, 100, 200)
    
    cv2.imshow('image',borde)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    resp    = np.matrix(gray)
    return resp.getA1()


directory = open("paths.txt", "r").read().split("\n")

X=[]
Y=[]

for idx_path,path in enumerate(directory):
    print(path)    
    for path_img in ls(path)[:10]:
        resp = preprocessingIMG(path_img)
        X.append(resp)
        
        tmp = np.zeros(len(directory))
        tmp[idx_path]=1
        Y.append(tmp)

X = np.matrix(X)
Y = np.matrix(Y)

np.save("X",X)
np.save("Y",Y)