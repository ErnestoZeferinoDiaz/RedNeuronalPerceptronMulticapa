from libs.libs import *
from libs.functions import *

directory = open("paths.txt", "r").read().split("\n")

#path = "../train/20.jpg"
#img  = cv2.imread(path)
url = "https://cnnespanol.cnn.com/wp-content/uploads/2020/07/200703104728-labrador-retriever-stock-super-169.jpg?quality=100&strip=info"
img  = url_to_image(url)

x    = preprocessingIMG(img)

print(x.size)

rows = x.shape[0]

r = RedNeuronal(
   [x.size,100,50,len(directory)],
   [sigmoid,sigmoid,sigmoid],
   0.01
)

r.loadModel("/content/drive/MyDrive/checkpoints")
y=r.frontPropagation(x)
m=y.max()
i=np.where(y==m)

print(y)
print(directory[i[0][0]])